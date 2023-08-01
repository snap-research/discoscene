# python3.7
"""Contains the base class for GAN-related metric computation.

Different from other deep models, evaluating a generative model (especially the
generator part) often requires to set up a collection of synthesized data beyond
the given validation set. For this purpose, it requires to sample a collection
of latent codes. To ensure the reproducibility as well as the evaluation
consistency during the training process, one may need to specify the latent code
collection and also save the collection used. Accordingly, this base class
handles the latent codes loading, splitting to replicas, and saving.
"""

import os.path
import numpy as np

import torch
import torch.nn.functional as F

from .base_metric import BaseMetric

__all__ = ['BaseGANMetric']


class BaseGANMetric(BaseMetric):
    """Defines the class for GAN-related metric computation.

    This class deals with the latent codes for image synthesising for the
    current replica. In this way, one can conduct evaluation with the same
    latent codes during the training process. If the latent codes are passed
    from outside, this class will do the split and only keep the ones used for
    the current replica. Meanwhile, this class will also collect the latent
    codes for all replicas together and save it to `work_dir` to ensure the
    reproducibility.
    """

    def __init__(self,
                 name=None,
                 work_dir=None,
                 logger=None,
                 tb_writer=None,
                 batch_size=1,
                 latent_num=-1,
                 latent_dim=512,
                 latent_codes=None,
                 label_dim=0,
                 labels=None,
                 seed=0):
        """Initialization with latent codes loading, splitting, and saving.

        Args:
            latent_num: Number of latent codes used for all replicas during
                evaluation. This field is required to be positive. (default: -1)
            latent_dim: The dimension of the latent space. (default: 512)
            latent_codes: The latent codes used for image generation. If this
                field is not provided, the latent codes will be randomly
                sampled. (default: None)
            label_dim: The dimension of labels, i.e., number of classes.
                (default: 0)
            labels: The labels used for image generation. If this field is not
                provided, the labels will be randomly sampled. (default: None)
            seed: Seed used for sampling. This is essential to ensure the
                reproducibility. (default: 0)
        """
        super().__init__(name, work_dir, logger, tb_writer, batch_size)
        assert latent_num > 0
        self.latent_num = latent_num
        self.seed = seed + self.rank
        self.replica_latent_num = self.get_replica_num(self.latent_num)
        self.replica_indices = self.get_indices(self.latent_num)
        assert self.replica_latent_num == len(self.replica_indices)
        # Path to save/load the latent codes and labels.
        self.latent_file = os.path.join(
            self.work_dir, f'{self.name}_latents_{self.latent_num}.npy')
        self.label_file = os.path.join(
            self.work_dir, f'{self.name}_labels_{self.latent_num}.npy')
        # Whether to use random latents/labels or use given latents/labels
        self.random_latents = (latent_codes is None)
        self.random_labels = (labels is None)
        self.prepare_latents(latent_dim, latent_codes)
        self.prepare_labels(label_dim, labels)

    def prepare_latents(self, latent_dim, latent_codes):
        """Prepares latent codes that will be used for evaluation."""
        # Get the dimension of latent space.
        if isinstance(latent_dim, int):
            assert latent_dim > 0
            latent_dim = (latent_dim,)
        assert isinstance(latent_dim, (list, tuple))
        self.latent_dim = tuple(latent_dim)

        # Load latent codes if needed.
        if isinstance(latent_codes, str) and os.path.exists(latent_codes):
            self.logger.info(f'Loading latent codes from `{latent_codes}` '
                             f'{self.log_tail}.',
                             is_verbose=True)
            file_ext = os.path.splitext(latent_codes)
            if file_ext in ['.pt', '.pth']:
                latent_codes = torch.load(latent_codes)
            elif file_ext in ['.npy', '.npz']:
                latent_codes = np.load(latent_codes)
            else:
                raise ValueError(f'Unknown file extension `{file_ext}`!')

        # Get latent codes for the current replica.
        self.logger.info(f'Preparing latent codes {self.log_tail}.',
                         is_verbose=True)
        if latent_codes is not None:
            assert not self.random_latents
            assert latent_codes.shape[0] > self.latent_num
            assert latent_codes.shape[1:] == self.latent_dim
            assert isinstance(latent_codes, (np.ndarray, torch.Tensor))
            latent_codes = latent_codes[:self.latent_num]
            latent_codes = latent_codes[self.replica_indices]
            if isinstance(latent_codes, np.ndarray):
                latent_codes = torch.from_numpy(latent_codes)
            latent_codes = latent_codes.to(torch.float32)
        else:
            assert self.random_latents
            g = torch.Generator(device=self.device)
            g.manual_seed(self.seed)

        # Save latent codes for reproducibility.
        self.logger.info(f'Saving the latent codes to `{self.latent_file}` '
                         f'{self.log_tail}.',
                         is_verbose=True)
        all_codes = []
        batch_size = self.batch_size
        for start in range(0, self.replica_latent_num, batch_size):
            if self.random_latents:
                batch_codes = torch.randn((batch_size, *self.latent_dim),
                                          generator=g, device=self.device)
            else:
                batch_codes = latent_codes[start:start + batch_size]
            gathered_codes = self.gather_batch_results(batch_codes)
            self.append_batch_results(gathered_codes, all_codes)
        all_codes = self.gather_all_results(all_codes)

        if self.is_chief:
            all_codes = all_codes[:self.latent_num]
            assert all_codes.shape == (self.latent_num, *self.latent_dim)
            np.save(self.latent_file, all_codes)
        else:
            assert len(all_codes) == 0
        self.sync()

    def prepare_labels(self, label_dim, labels):
        """Prepares labels that will be used for evaluation."""
        # Get label dimension.
        if label_dim is None:
            label_dim = 0
        assert isinstance(label_dim, int) and label_dim >= 0
        self.label_dim = label_dim

        # Load labels if needed.
        if isinstance(labels, str) and os.path.exists(labels):
            self.logger.info(f'Loading labels from `{labels}` {self.log_tail}.',
                             is_verbose=True)
            file_ext = os.path.splitext(labels)
            if file_ext in ['.pt', '.pth']:
                labels = torch.load(labels)
            elif file_ext in ['.npy', '.npz']:
                labels = np.load(labels)
            else:
                raise ValueError(f'Unknown file extension `{file_ext}`!')

        # Get labels for the current replica.
        self.logger.info(f'Preparing labels for each replica {self.log_tail}.',
                         is_verbose=True)
        if labels is not None:
            assert not self.random_labels
            assert labels.ndim == 2
            assert labels.shape[0] > self.latent_num
            assert labels.shape[1] == self.label_dim
            assert isinstance(labels, (np.ndarray, torch.Tensor))
            labels = labels[:self.latent_num]
            labels = labels[self.replica_indices]
            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels)
            labels = labels.to(torch.float32)
        else:
            assert self.random_labels
            g = torch.Generator(device=self.device)
            g.manual_seed(self.seed)

        # Save labels for reproducibility.
        self.logger.info(f'Saving the labels to `{self.label_file}` '
                         f'{self.log_tail}.',
                         is_verbose=True)
        all_labels = []
        batch_size = self.batch_size
        for start in range(0, self.replica_latent_num, batch_size):
            if self.random_labels:
                if self.label_dim == 0:
                    batch_labels = torch.zeros((batch_size, 0),
                                               device=self.device)
                else:
                    rnd_labels = torch.randint(
                        low=0, high=self.label_dim, size=(batch_size,),
                        generator=g, device=self.device)
                    batch_labels = F.one_hot(
                        rnd_labels, num_classes=self.label_dim)
            else:
                batch_labels = labels[start:start + batch_size]
            gathered_labels = self.gather_batch_results(batch_labels)
            self.append_batch_results(gathered_labels, all_labels)
        all_labels = self.gather_all_results(all_labels)

        if self.is_chief:
            all_labels = all_labels[:self.latent_num]
            assert all_labels.shape == (self.latent_num, self.label_dim)
            np.save(self.label_file, all_labels)
        else:
            assert len(all_labels) == 0
        self.sync()

    def evaluate(self, *args):
        raise NotImplementedError('Should be implemented in derived class!')

    def _is_better_than(self, metric_name, new, ref):
        raise NotImplementedError('Should be implemented in derived classes '
                                  'for evaluation results comparison!')

    def save(self, result, target_filename=None, log_suffix=None, tag=None):
        raise NotImplementedError('Should be implemented in derived class!')

    def info(self):
        metric_info = super().info()
        metric_info['Num latents'] = self.latent_num
        metric_info['Latent dimension'] = self.latent_dim
        metric_info['Label dimension'] = self.label_dim
        metric_info['Random seed'] = self.seed
        return metric_info
