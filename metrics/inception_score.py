# python3.7
"""Contains the class to evaluate GANs with Inception Score (IS).

IS metric is introduced in paper

https://proceedings.neurips.cc/paper/2016/file/8a3363abe792db2d8761d6403605aeb7-Paper.pdf
"""

import os.path
import time
import numpy as np

import torch
import torch.nn.functional as F

from models import build_model
from .base_gan_metric import BaseGANMetric
from .utils import compute_is
from utils.misc import gather_data

__all__ = ['ISMetric', 'IS50K']

PROBS_DIM = 1008  # Dimension of predicted probabilities.


class ISMetric(BaseGANMetric):
    """Defines the class for IS metric computation."""

    def __init__(self,
                 name='IS',
                 work_dir=None,
                 logger=None,
                 tb_writer=None,
                 batch_size=1,
                 latent_num=-1,
                 latent_dim=512,
                 latent_codes=None,
                 label_dim=0,
                 labels=None,
                 seed=0,
                 num_splits=10,
                 bbox_as_input=False):
        """Initializes the class with number of latents and collection splits.

        Args:
            latent_num: Name of latent codes used for evaluation.
            num_splits: Number of splits of the entire random collection. The
                KL-divergence will be computed within each split. Then the
                mean and variance will be computed across different splits.
                (default: 10)
        """
        super().__init__(name=name,
                         work_dir=work_dir,
                         logger=logger,
                         tb_writer=tb_writer,
                         batch_size=batch_size,
                         latent_num=latent_num,
                         latent_dim=latent_dim,
                         latent_codes=latent_codes,
                         label_dim=label_dim,
                         labels=labels,
                         seed=seed)
        self.num_splits = num_splits
        self.bbox_as_input = bbox_as_input

        # Build inception model for feature extraction.
        self.inception_model = build_model('InceptionModel', align_tf=True)

    def extract_fake_probs(self, generator, generator_kwargs, data_loader):
        """Extracts inception predictions from fake data."""
        latent_num = self.latent_num
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

        self.logger.info(f'Extracting inception predictions from fake data '
                         f'{self.log_tail}.',
                         is_verbose=True)
        self.logger.init_pbar()
        pbar_task = self.logger.add_pbar_task('Fake', total=latent_num)
        all_probs = []
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
                if self.bbox_as_input:
                    bbox_kwargs = gather_data([data_loader.dataset.get_bbox(np.random.randint(len(data_loader.dataset))) for i in range(batch_codes.shape[0])], device=batch_codes.device)
                    batch_images = G(batch_codes, batch_labels, bbox_kwargs=bbox_kwargs, **G_kwargs)['image']
                else:
                    batch_images = G(batch_codes, batch_labels, **G_kwargs)['image']
                batch_probs = self.inception_model(batch_images,
                                                   output_predictions=True,
                                                   remove_logits_bias=True)
                gathered_probs = self.gather_batch_results(batch_probs)
                self.append_batch_results(gathered_probs, all_probs)
            self.logger.update_pbar(pbar_task, (end - start) * self.world_size)
        self.logger.close_pbar()
        all_probs = self.gather_all_results(all_probs)[:latent_num]

        if self.is_chief:
            assert all_probs.shape == (latent_num, PROBS_DIM)
        else:
            assert len(all_probs) == 0
            all_probs = None

        if G_mode:
            G.train()  # restore model training mode.

        self.sync()
        return all_probs

    def evaluate(self, _data_loader, generator, generator_kwargs):
        probs = self.extract_fake_probs(generator, generator_kwargs, _data_loader)
        if self.is_chief:
            is_mean, is_std = compute_is(probs, self.num_splits)
            result = {f'{self.name}_mean': is_mean, f'{self.name}_std': is_std}
        else:
            assert probs is None
            result = None
        self.sync()
        return result

    def _is_better_than(self, metric_name, new, ref):
        """Higher inception score mean is better."""
        if metric_name == f'{self.name}_mean':
            return ref is None or new > ref
        return None

    def save(self, result, target_filename=None, log_suffix=None, tag=None):
        if not self.is_chief:
            assert result is None
            self.sync()
            return

        assert isinstance(result, dict)
        is_mean = result[f'{self.name}_mean']
        is_std = result[f'{self.name}_std']
        assert isinstance(is_mean, float) and isinstance(is_std, float)
        prefix = f'Evaluating `{self.name}`: '
        if log_suffix is None:
            msg = f'{prefix}mean {is_mean:.3f}, std {is_std:.3f}.'
        else:
            msg = f'{prefix}mean {is_mean:.3f}, std {is_std:.3f}, {log_suffix}.'
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
        metric_info['Num splits for testing'] = self.num_splits
        return metric_info


class IS50K(ISMetric):
    """Defines the class for IS50K metric computation.

    50_000 random samples will be used for evaluation.
    """

    def __init__(self,
                 name='IS50K',
                 work_dir=None,
                 logger=None,
                 tb_writer=None,
                 batch_size=1,
                 latent_dim=512,
                 latent_codes=None,
                 label_dim=0,
                 labels=None,
                 seed=0,
                 num_splits=10):
        super().__init__(name=name,
                         work_dir=work_dir,
                         logger=logger,
                         tb_writer=tb_writer,
                         batch_size=batch_size,
                         latent_num=50_000,
                         latent_dim=latent_dim,
                         latent_codes=latent_codes,
                         label_dim=label_dim,
                         labels=labels,
                         seed=seed,
                         num_splits=num_splits)
