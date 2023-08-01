# python3.7
"""Contains the class to evaluate GANs with precision and recall.

Precision-Recall metric is introduced in paper

https://arxiv.org/pdf/1904.06991.pdf
"""

import os.path
import time
import numpy as np

import torch
import torch.nn.functional as F

from models import build_model
from utils.misc import get_cache_dir
from .base_gan_metric import BaseGANMetric
from .utils import compute_gan_precision_recall

__all__ = ['GANPRMetric', 'GANPR50K', 'GANPR50KFull']

FEATURE_DIM = 4096  # Dimension of perceptual feature from VGG16.


class GANPRMetric(BaseGANMetric):
    """Defines the class for precision-recall metric for GAN evaluation."""

    def __init__(self,
                 name='GANPR',
                 work_dir=None,
                 logger=None,
                 tb_writer=None,
                 batch_size=1,
                 latent_dim=512,
                 latent_codes=None,
                 label_dim=0,
                 labels=None,
                 seed=0,
                 real_num=-1,
                 fake_num=-1,
                 chunk_size=10000,
                 top_k=3):
        """Initializes the class with number of real/fakes samples for GANPR.

        Args:
            real_num: Number of real images used for GANPR evaluation. If not
                set, all images from the given evaluation dataset will be used.
                (default: -1)
            fake_num: Number of fake images used for GANPR evaluation.
                (default: -1)
            chunk_size: Number of samples for each computation chunk, which will
                save memory. (default: 10000)
            top_k: Hyper-parameter for precision-recall computation.
                (default: 3)
        """
        super().__init__(name=name,
                         work_dir=work_dir,
                         logger=logger,
                         tb_writer=tb_writer,
                         batch_size=batch_size,
                         latent_num=fake_num,
                         latent_dim=latent_dim,
                         latent_codes=latent_codes,
                         label_dim=label_dim,
                         labels=labels,
                         seed=seed)
        self.real_num = real_num
        self.fake_num = fake_num
        self.chunk_size = chunk_size
        self.top_k = top_k

        # Build perceptual model for feature extraction.
        self.perceptual_model = build_model(
            'PerceptualModel', no_top=False, enable_lpips=False)

    def extract_real_features(self, data_loader):
        """Extracts perceptual features from real data."""
        if self.real_num < 0:
            real_num = len(data_loader.dataset)
        else:
            real_num = min(self.real_num, len(data_loader.dataset))

        dataset_name = os.path.splitext(
            os.path.basename(data_loader.dataset.root_dir))[0]
        cache_name = f'{dataset_name}_{real_num}_perceptual_feature.npy'
        cache_path = os.path.join(get_cache_dir(), cache_name)

        if os.path.exists(cache_path):
            self.logger.info(f'Loading statistics of real data from cache '
                             f'`{cache_path}` {self.log_tail}.')
            all_features = np.load(cache_path) if self.is_chief else None
            self.sync()
            return all_features

        self.logger.info(f'Extracting perceptual features from real data '
                         f'{self.log_tail}.',
                         is_verbose=True)
        self.logger.init_pbar()
        pbar_task = self.logger.add_pbar_task('Real', total=real_num)
        all_features = []
        batch_size = data_loader.batch_size
        replica_num = self.get_replica_num(real_num)
        for batch_idx in range(len(data_loader)):
            if batch_idx * batch_size >= replica_num:
                # NOTE: Here, we always go through the entire dataset to make
                # sure the next evaluator can visit the data loader from the
                # beginning.
                _batch_data = next(data_loader)
                continue
            with torch.no_grad():
                batch_data = next(data_loader)['image'].cuda().detach()
                batch_features = self.perceptual_model(
                    batch_data, resize_input=True, return_tensor='feature')
                gathered_features = self.gather_batch_results(batch_features)
                self.append_batch_results(gathered_features, all_features)
            self.logger.update_pbar(pbar_task, batch_size * self.world_size)
        self.logger.close_pbar()
        all_features = self.gather_all_results(all_features)[:real_num]

        self.logger.info(f'Saving statistics of real data to cache '
                         f'`{cache_path}` {self.log_tail}.')
        if self.is_chief:
            assert all_features.shape == (real_num, FEATURE_DIM)
            np.save(cache_path, all_features)
        else:
            assert len(all_features) == 0
            all_features = None
        self.sync()
        return all_features

    def extract_fake_features(self, generator, generator_kwargs):
        """Extracts perceptual features from fake data."""
        fake_num = self.fake_num
        batch_size = self.batch_size
        if self.random_latents:
            g1 = torch.Generator(device=self.device)
            g1.manual_seed(self.seed)
        else:
            latent_codes = np.load(self.latent_file)[self.replica_indices]
            latent_codes = torch.from_numpy(latent_codes).to(torch.float32)
        if self.random_labels:
            g2 = torch.Generator(device=self.device)
            g2.manual_seed(self.seed)
        else:
            labels = np.load(self.label_file)[self.replica_indices]
            labels = torch.from_numpy(labels).to(torch.float32)

        G = generator
        G_kwargs = generator_kwargs
        G_mode = G.training  # save model training mode.
        G.eval()

        self.logger.info(f'Extracting perceptual features from fake data '
                         f'{self.log_tail}.',
                         is_verbose=True)
        self.logger.init_pbar()
        pbar_task = self.logger.add_pbar_task('Fake', total=fake_num)
        all_features = []
        for start in range(0, self.replica_latent_num, batch_size):
            end = min(start + batch_size, self.replica_latent_num)
            with torch.no_grad():
                if self.random_latents:
                    batch_codes = torch.randn((end - start, *self.latent_dim),
                                              generator=g1, device=self.device)
                else:
                    batch_codes = latent_codes[start:end].cuda().detach()
                if self.random_labels:
                    if self.label_dim == 0:
                        batch_labels = torch.zeros((end - start, 0),
                                                   device=self.device)
                    else:
                        rnd_labels = torch.randint(
                            low=0, high=self.label_dim, size=(end - start,),
                            generator=g2, device=self.device)
                        batch_labels = F.one_hot(
                            rnd_labels, num_classes=self.label_dim)
                else:
                    batch_labels = labels[start:end].cuda().detach()
                batch_images = G(batch_codes, batch_labels, **G_kwargs)['image']
                batch_features = self.perceptual_model(
                    batch_images, resize_input=True, return_tensor='feature')
                gathered_features = self.gather_batch_results(batch_features)
                self.append_batch_results(gathered_features, all_features)
            self.logger.update_pbar(pbar_task, (end - start) * self.world_size)
        self.logger.close_pbar()
        all_features = self.gather_all_results(all_features)[:fake_num]

        if self.is_chief:
            assert all_features.shape == (fake_num, FEATURE_DIM)
        else:
            assert len(all_features) == 0
            all_features = None

        if G_mode:
            G.train()  # restore model training mode.

        self.sync()
        return all_features

    def evaluate(self, data_loader, generator, generator_kwargs):
        real_features = self.extract_real_features(data_loader)
        fake_features = self.extract_fake_features(generator, generator_kwargs)
        if self.is_chief:
            precision, recall = compute_gan_precision_recall(
                fake_features, real_features, self.chunk_size, self.top_k)
            result = {
                f'{self.name}_precision': precision,
                f'{self.name}_recall': recall
            }
        else:
            assert real_features is None and fake_features is None
            result = None
        self.sync()
        return result

    def _is_better_than(self, metric_name, new, ref):
        """Performance comparison.

        Higher precision means higher quality.
        Higher recall means higher diversity.
        """
        if metric_name == f'{self.name}_precision':
            return ref is None or new > ref
        if metric_name == f'{self.name}_recall':
            return ref is None or new > ref
        return None

    def save(self, result, target_filename=None, log_suffix=None, tag=None):
        if not self.is_chief:
            assert result is None
            self.sync()
            return

        assert isinstance(result, dict)
        is_mean = result[f'{self.name}_precision']
        is_std = result[f'{self.name}_recall']
        assert isinstance(is_mean, float) and isinstance(is_std, float)
        prefix = f'Evaluating `{self.name}`: '
        if log_suffix is None:
            msg = f'{prefix}precision {is_mean:.3f}, recall {is_std:.3f}.'
        else:
            msg = (f'{prefix}precision {is_mean:.3f}, recall {is_std:.3f}, '
                   f'{log_suffix}.')
        self.logger.info(msg)

        with open(os.path.join(self.work_dir, f'{self.name}.txt'), 'a+') as f:
            date = time.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f'[{date}] {msg}\n')

        # Save to TensorBoard if needed.
        if self.tb_writer is not None:
            if tag is None:
                self.logger.warning('`Tag` is missing when writing data to '
                                    'TensorBoard, hence, the data may be mixed '
                                    'up!')
            self.tb_writer.add_scalar(f'Metrics/{self.name}_mean', is_mean, tag)
            self.tb_writer.add_scalar(f'Metrics/{self.name}_std', is_std, tag)
            self.tb_writer.flush()
        self.sync()

    def info(self):
        metric_info = super().info()
        metric_info['Num real samples'] = self.real_num
        metric_info['Num fake samples'] = self.fake_num
        metric_info['Chuck size for computation'] = self.chunk_size
        metric_info['Top-k for positive hitting'] = self.top_k
        return metric_info


class GANPR50K(GANPRMetric):
    """Defines the class for GANPR50K metric computation.

    50_000 real/fake samples will be used for feature extraction.
    """

    def __init__(self,
                 name='GANPR50K',
                 work_dir=None,
                 logger=None,
                 tb_writer=None,
                 batch_size=1,
                 latent_dim=512,
                 latent_codes=None,
                 label_dim=0,
                 labels=None,
                 seed=0,
                 chunk_size=10000,
                 top_k=3):
        super().__init__(name=name,
                         work_dir=work_dir,
                         logger=logger,
                         tb_writer=tb_writer,
                         batch_size=batch_size,
                         latent_dim=latent_dim,
                         latent_codes=latent_codes,
                         label_dim=label_dim,
                         labels=labels,
                         seed=seed,
                         real_num=50_000,
                         fake_num=50_000,
                         chunk_size=chunk_size,
                         top_k=top_k)


class GANPR50KFull(GANPRMetric):
    """Defines the class for GANPR50KFull metric computation.

    50_000 fake samples and ALL (maximum 200_000) real samples will be used for
    feature extraction.
    """

    def __init__(self,
                 name='GANPR50KFull',
                 work_dir=None,
                 logger=None,
                 tb_writer=None,
                 batch_size=1,
                 latent_dim=512,
                 latent_codes=None,
                 label_dim=0,
                 labels=None,
                 seed=0,
                 chunk_size=10000,
                 top_k=3):
        super().__init__(name=name,
                         work_dir=work_dir,
                         logger=logger,
                         tb_writer=tb_writer,
                         batch_size=batch_size,
                         latent_dim=latent_dim,
                         latent_codes=latent_codes,
                         label_dim=label_dim,
                         labels=labels,
                         seed=seed,
                         real_num=200_000,
                         fake_num=50_000,
                         chunk_size=chunk_size,
                         top_k=top_k)
