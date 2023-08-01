# python3.7
"""Class to evaluate GANs with intra-class Frechet Inception Distance (ICFID).

FID metric is introduced in paper https://arxiv.org/pdf/1706.08500.pdf
"""

import os.path
import time
import numpy as np

import torch
import torch.nn.functional as F

from models import build_model
from utils.misc import get_cache_dir
from .base_gan_metric import BaseGANMetric
from .utils import compute_fid_from_feature

__all__ = ['ICFIDMetric', 'ICFID50K', 'ICFID50KFull']

FEATURE_DIM = 2048  # Dimension of inception feature.


class ICFIDMetric(BaseGANMetric):
    """Defines the class for intra-class FID (ICFID) metric computation."""

    def __init__(self,
                 name='ICFID',
                 work_dir=None,
                 logger=None,
                 tb_writer=None,
                 batch_size=1,
                 latent_dim=512,
                 latent_codes=None,
                 label_dim=0,
                 labels=None,
                 seed=0,
                 interested_classes=None,
                 real_num_per_cls=-1,
                 fake_num_per_cls=-1):
        """Initializes the class with number of real/fakes samples for ICFID.

        Args:
            interested_classes: A list of interested classes. If left empty, all
                available classes will be treated as interested. (default: None)
            real_num_per_cls: Number of real images in each class used for FID
                evaluation. If set to a number greater than the actual available
                amount of some classes, the metric will use all available data
                without padding. If not set, all images from the given class in
                evaluation dataset will be used. (default: -1)
            fake_num_per_cls: Number of fake images in each class used for FID
                evaluation. (default: -1)

        NOTE:
            `labels` is not used since to-be-evaluated classes are determined
                by `interested_classes`. Therefore, the labels will not be
                randomly sampled and `self.random_labels` is unavailable.
        """
        super().__init__(name=name,
                         work_dir=work_dir,
                         logger=logger,
                         tb_writer=tb_writer,
                         batch_size=batch_size,
                         latent_num=fake_num_per_cls,
                         latent_dim=latent_dim,
                         latent_codes=latent_codes,
                         label_dim=label_dim,
                         labels=None,
                         seed=seed)
        if labels is not None:
            self.logger.warning('`labels` is ignored in the calculation of '
                                'intra-class FID!')

        # Prepare labels for interested classes.
        if interested_classes is None:  # Evaluate for all classes.
            interested_classes = tuple(range(label_dim))
        assert isinstance(interested_classes, (tuple, list))

        self.interested_classes = interested_classes
        self.real_num_per_cls = real_num_per_cls
        self.fake_num_per_cls = fake_num_per_cls

        # Build inception model for feature extraction.
        self.inception_model = build_model('InceptionModel', align_tf=True)

    def prepare_labels(self, label_dim, labels):
        """Overrides the parent method to disable preparing labels randomly."""
        self.label_dim = label_dim
        self.random_labels = False

    def extract_real_features(self, data_loader):
        """Extracts inception features from real data."""
        # Load real features cache if available.
        dataset_name = os.path.splitext(
            os.path.basename(data_loader.dataset.root_dir))[0]
        if self.real_num_per_cls <= 0:
            cache_name = (f'{dataset_name}-label{self.interested_classes}-'
                          f'all-inception_features.npy')
        else:
            cache_name = (f'{dataset_name}-label{self.interested_classes}-'
                          f'{self.real_num_per_cls}each-inception_features.npy')
        cache_path = os.path.join(get_cache_dir(), cache_name.replace(' ', ''))

        if os.path.exists(cache_path):
            self.logger.info(f'Loading statistics of real data from cache '
                             f'`{cache_path}` {self.log_tail}.')
            if self.is_chief:
                feature_dict = np.load(cache_path, allow_pickle=True).item()
            else:
                feature_dict = None
            self.sync()
            return feature_dict

        dataset = data_loader.dataset
        if dataset.mirror:
            raise ValueError('Validation dataset should not be mirrored!')
        if data_loader.shuffle:
            raise ValueError('Validation dataset should not be shuffled!')

        # The following process will traverse all dataset items to finalize the
        # exact number for each class. Meanwhile, this process will also affirm
        # an early stopping point such that, once all interested classes have
        # got enough samples, the remaining samples within the dataset will not
        # be used anymore, which intends to reduce memory footprint and
        # processing time.

        # `ends_at` traces stopping points in the dataset. Each stopping point
        # marks the index where an interested class is sufficiently sampled OR
        # the index of the last sample of an interested class.
        ends_at = {cls_id: 0 for cls_id in self.interested_classes}

        # `cut_offs` traces the exact amount for each interested class.
        cut_offs = {cls_id: 0 for cls_id in self.interested_classes}

        # `to_targets` traces the number of samples required to reach the
        # target. This field only takes effect if `real_num_per_cls` is valid.
        # Otherwise, there will be no target because all samples will be used.
        if self.real_num_per_cls > 0:
            target = self.real_num_per_cls
            to_targets = {cls_id: target for cls_id in self.interested_classes}
            total_remaining = target * len(self.interested_classes)

        # Traverse the dataset items.
        for idx, item in enumerate(dataset.items[:len(dataset)], start=1):
            cls_id = item[1]
            if cls_id not in self.interested_classes:  # Not interested classes.
                continue
            if self.real_num_per_cls <= 0:  # Use all samples.
                cut_offs[cls_id] += 1
                ends_at[cls_id] = idx
                continue
            # Determine early stopping.
            if to_targets[cls_id] > 0:  # Class with insufficient samples.
                cut_offs[cls_id] += 1
                ends_at[cls_id] = idx
                to_targets[cls_id] -= 1
                total_remaining -= 1
                if total_remaining == 0:
                    # Stop traversing if already having enough samples.
                    break

        # Get the amount of real samples need to be processed by each replica.
        replica_real_num = self.get_replica_num(max(ends_at.values()))

        # Prepare real samples.
        self.logger.info(f'Extracting inception features from real data '
                         f'{self.log_tail}.',
                         is_verbose=True)
        self.logger.init_pbar()
        pbar_task = self.logger.add_pbar_task('Real', total=len(data_loader))
        feature_dict = {cls_id: list() for cls_id in self.interested_classes}
        batch_size = data_loader.batch_size
        for batch_idx in range(len(data_loader)):
            if batch_idx * batch_size >= replica_real_num:
                # NOTE: Here, we always go through the entire dataset to make
                # sure the next evaluator can visit the data loader from the
                # beginning.
                _batch_data = next(data_loader)
                continue
            with torch.no_grad():
                batch_data = next(data_loader)
                batch_images = batch_data['image'].cuda().detach()
                batch_labels = batch_data['raw_label']
                batch_features = self.inception_model(batch_images)
                gathered_labels = self.gather_batch_results(batch_labels)
                gathered_features = self.gather_batch_results(batch_features)
                if not self.is_chief:  # Skip if not chief.
                    assert not gathered_features and not gathered_labels
                    continue
                for feature, cls_id in zip(gathered_features, gathered_labels):
                    if cls_id not in self.interested_classes:
                        continue
                    if len(feature_dict[cls_id]) < cut_offs[cls_id]:
                        feature_dict[cls_id].append(feature[np.newaxis, :])
            self.logger.update_pbar(pbar_task, batch_size * self.world_size)
        self.logger.close_pbar()
        for cls_id in self.interested_classes:
            feature_dict[cls_id] = self.gather_all_results(feature_dict[cls_id])

        # Save cache.
        self.logger.info(f'Saving statistics of real data to cache '
                         f'`{cache_path}` {self.log_tail}.')
        if self.is_chief:
            assert all(
                feature_dict[cls_id].shape == (cut_offs[cls_id], FEATURE_DIM)
                for cls_id in self.interested_classes)
            np.save(cache_path, feature_dict)
        else:
            assert all(len(feature) == 0 for feature in feature_dict.values())
            feature_dict = None
        self.sync()
        return feature_dict

    def extract_fake_features(self, generator, generator_kwargs, cls_id):
        """Extracts inception features from fake data for a specific class.

        Args:
            generator: The generator network used to generate fake images for
                feature extraction.
            generator_kwargs: Runtime keyword arguments of generator network.
            cls_id: Class index as the condition for synthesis.

        NOTE: By default, the same latent codes will be used for each class.
        """
        fake_num_per_cls = self.fake_num_per_cls
        batch_size = self.batch_size
        if self.random_latents:
            g = torch.Generator(device=self.device)
            g.manual_seed(self.seed)
        else:
            latent_codes = np.load(self.latent_file)[self.replica_indices]
            latent_codes = torch.from_numpy(latent_codes).to(torch.float32)
        label = F.one_hot(torch.as_tensor(cls_id, device=self.device),
                          num_classes=self.label_dim)

        G = generator
        G_kwargs = generator_kwargs
        G_mode = G.training  # save model training mode.
        G.eval()

        self.logger.info(f'Extracting inception features from fake data '
                         f'{self.log_tail}.',
                         is_verbose=True)
        self.logger.init_pbar()
        pbar_task = self.logger.add_pbar_task(f'Fake class{cls_id}',
                                              total=fake_num_per_cls)
        all_features = []
        for start in range(0, self.replica_latent_num, batch_size):
            end = min(start + batch_size, self.replica_latent_num)
            actual_size = end - start
            with torch.no_grad():
                if self.random_latents:
                    batch_codes = torch.randn((actual_size, *self.latent_dim),
                                              generator=g,
                                              device=self.device)
                else:
                    batch_codes = latent_codes[start:end].cuda().detach()
                batch_labels = label.repeat(actual_size, 1).detach()
                batch_images = G(batch_codes, batch_labels, **G_kwargs)['image']
                batch_features = self.inception_model(batch_images)
                gathered_features = self.gather_batch_results(batch_features)
                self.append_batch_results(gathered_features, all_features)
            self.logger.update_pbar(pbar_task, actual_size * self.world_size)
        self.logger.close_pbar()
        all_features = self.gather_all_results(all_features)[:fake_num_per_cls]

        if self.is_chief:
            assert all_features.shape == (fake_num_per_cls, FEATURE_DIM)
        else:
            assert len(all_features) == 0
            all_features = None

        if G_mode:
            G.train()  # restore model training mode.

        self.sync()
        return all_features

    def evaluate(self, data_loader, generator, generator_kwargs):
        real_features = self.extract_real_features(data_loader)

        ic_fids = dict()
        # Calculate FID for each class.
        for cls_id in self.interested_classes:
            fake_features = self.extract_fake_features(
                generator, generator_kwargs, cls_id)
            if self.is_chief:
                ic_fids[cls_id] = compute_fid_from_feature(
                    fake_features, real_features[cls_id])
            else:
                assert fake_features is None
        if self.is_chief:
            avg_ic_fid = np.mean(list(ic_fids.values()))
            result = {
                f'{self.name}_individual': ic_fids,
                f'{self.name}_avg': avg_ic_fid
            }
        else:
            assert real_features is None
            assert len(ic_fids) == 0
            result = None
        self.sync()
        return result

    def _is_better_than(self, metric_name, new, ref):
        """Lower average intra-class FID is better."""
        if metric_name == f'{self.name}_avg':
            return ref is None or new < ref
        return None

    def save(self, result, target_filename=None, log_suffix=None, tag=None):
        if not self.is_chief:
            assert result is None
            self.sync()
            return

        assert isinstance(result, dict)
        ic_fids = result[f'{self.name}_individual']
        avg_ic_fid = result[f'{self.name}_avg']
        assert isinstance(ic_fids, dict)

        prefix = f'Evaluating `{self.name}` on {len(ic_fids)} classes:'
        msg = f'{prefix} Average: {avg_ic_fid:.3f}'
        for raw_label, ic_fid in ic_fids.items():
            msg = f'{msg}, Class{raw_label}: {ic_fid:.3f}'
        if log_suffix is not None:
            msg = f'{msg}, {log_suffix}.'
        self.logger.info(msg)

        with open(os.path.join(self.work_dir, f'{self.name}.txt'), 'a+') as f:
            date = time.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f'[{date}] {msg}\n')

        # Save to TensorBoard if needed.
        if self.tb_writer is not None:
            if tag is None:
                self.logger.warning('`Tag` is missing when writing data to '
                                    'TensorBoard, hence, the data may be '
                                    'mixed up!')
            self.tb_writer.add_scalar(f'Metrics/{self.name}_avg',
                                      avg_ic_fid, tag)
            self.tb_writer.add_scalars(f'Metrics/{self.name}_individual',
                                       {f'Class{k}': v
                                        for k, v in ic_fids.items()}, tag)
            self.tb_writer.flush()
        self.sync()

    def info(self):
        metric_info = super().info()
        metric_info['Interested classes'] = self.interested_classes
        metric_info['Num real samples per class'] = self.real_num_per_cls
        metric_info['Num fake samples per class'] = self.fake_num_per_cls
        return metric_info


class ICFID50K(ICFIDMetric):
    """Defines the class for ICFID50K metric computation.

    50_000 real/fake samples per class will be used for feature extraction.
    """

    def __init__(self,
                 name='ICFID50K',
                 work_dir=None,
                 logger=None,
                 tb_writer=None,
                 batch_size=1,
                 latent_dim=512,
                 latent_codes=None,
                 label_dim=0,
                 interested_classes=None,
                 seed=0):
        super().__init__(name=name,
                         work_dir=work_dir,
                         logger=logger,
                         tb_writer=tb_writer,
                         batch_size=batch_size,
                         latent_dim=latent_dim,
                         latent_codes=latent_codes,
                         label_dim=label_dim,
                         interested_classes=interested_classes,
                         seed=seed,
                         real_num_per_cls=50_000,
                         fake_num_per_cls=50_000)


class ICFID50KFull(ICFIDMetric):
    """Defines the class for ICFID50KFull metric computation.

    50_000 fake samples per class and ALL real samples will be used for feature
    extraction.
    """

    def __init__(self,
                 name='ICFID50KFull',
                 work_dir=None,
                 logger=None,
                 tb_writer=None,
                 batch_size=1,
                 latent_dim=512,
                 latent_codes=None,
                 label_dim=0,
                 interested_classes=None,
                 seed=0):
        super().__init__(name=name,
                         work_dir=work_dir,
                         logger=logger,
                         tb_writer=tb_writer,
                         batch_size=batch_size,
                         latent_dim=latent_dim,
                         latent_codes=latent_codes,
                         label_dim=label_dim,
                         interested_classes=interested_classes,
                         seed=seed,
                         real_num_per_cls=-1,
                         fake_num_per_cls=50_000)
