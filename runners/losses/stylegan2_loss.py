# python3.7
"""Defines loss functions for StyleGAN2 training."""

import numpy as np

import torch
import torch.nn.functional as F

from third_party.stylegan2_official_ops import conv2d_gradfix
from utils.dist_utils import ddp_sync
from .base_loss import BaseLoss

__all__ = ['StyleGAN2Loss']


class StyleGAN2Loss(BaseLoss):
    """Contains the class to compute losses for training StyleGAN2.

    Basically, this class contains the computation of adversarial loss for both
    generator and discriminator, perceptual path length regularization for
    generator, and gradient penalty as the regularization for discriminator.
    """

    def __init__(self, runner, d_loss_kwargs=None, g_loss_kwargs=None):
        """Initializes with models and arguments for computing losses."""

        if runner.enable_amp:
            raise NotImplementedError('StyleGAN2 loss does not support '
                                      'automatic mixed precision training yet.')

        # Setting for discriminator loss.
        self.d_loss_kwargs = d_loss_kwargs or dict()
        # Loss weight for gradient penalty on real images.
        self.r1_gamma = self.d_loss_kwargs.get('r1_gamma', 10.0)
        # How often to perform gradient penalty regularization.
        self.r1_interval = self.d_loss_kwargs.get('r1_interval', 16)

        if self.r1_interval is None or self.r1_interval <= 0:
            self.r1_interval = 1
            self.r1_gamma = 0.0
        self.r1_interval = int(self.r1_interval)
        assert self.r1_gamma >= 0.0
        runner.running_stats.add('Loss/D Fake',
                                 log_name='loss_d_fake',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')
        runner.running_stats.add('Loss/D Real',
                                 log_name='loss_d_real',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')
        if self.r1_gamma > 0.0:
            runner.running_stats.add('Loss/Real Gradient Penalty',
                                     log_name='loss_gp',
                                     log_format='.1e',
                                     log_strategy='AVERAGE')

        # Settings for generator loss.
        self.g_loss_kwargs = g_loss_kwargs or dict()
        # Factor to shrink the batch size for path length regularization.
        self.pl_batch_shrink = int(self.g_loss_kwargs.get('pl_batch_shrink', 2))
        # Loss weight for perceptual path length regularization.
        self.pl_weight = self.g_loss_kwargs.get('pl_weight', 2.0)
        # Decay factor for perceptual path length regularization.
        self.pl_decay = self.g_loss_kwargs.get('pl_decay', 0.01)
        # How often to perform perceptual path length regularization.
        self.pl_interval = self.g_loss_kwargs.get('pl_interval', 4)

        if self.pl_interval is None or self.pl_interval <= 0:
            self.pl_interval = 1
            self.pl_weight = 0.0
        self.pl_interval = int(self.pl_interval)
        assert self.pl_batch_shrink >= 1
        assert self.pl_weight >= 0.0
        assert 0.0 <= self.pl_decay <= 1.0
        runner.running_stats.add('Loss/G',
                                 log_name='loss_g',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')
        if self.pl_weight > 0.0:
            runner.running_stats.add('Loss/Path Length Penalty',
                                     log_name='loss_pl',
                                     log_format='.1e',
                                     log_strategy='AVERAGE')
            self.pl_mean = torch.zeros((), device=runner.device)

        # Log loss settings.
        runner.logger.info('gradient penalty (D regularizer):', indent_level=1)
        runner.logger.info(f'r1_gamma: {self.r1_gamma}', indent_level=2)
        runner.logger.info(f'r1_interval: {self.r1_interval}', indent_level=2)
        runner.logger.info('perceptual path length penalty (G regularizer):',
                           indent_level=1)
        runner.logger.info(f'pl_batch_shrink: {self.pl_batch_shrink}',
                           indent_level=2)
        runner.logger.info(f'pl_weight: {self.pl_weight}', indent_level=2)
        runner.logger.info(f'pl_decay: {self.pl_decay}', indent_level=2)
        runner.logger.info(f'pl_interval: {self.pl_interval}', indent_level=2)

    @staticmethod
    def run_G(runner, batch_size=None, sync=True, requires_grad=False):
        """Forwards generator.

        NOTE: The flag `requires_grad` sets whether to compute the gradient for
            latent z. When computing the `pl_penalty` with part of the generator
            frozen (e.g., mapping network), this flag should be set to `True` to
            retain the computation graph.
        """
        # Prepare latent codes and labels.
        batch_size = batch_size or runner.batch_size
        latent_dim = runner.models['generator'].latent_dim
        label_dim = runner.models['generator'].label_dim
        latents = torch.randn((batch_size, *latent_dim),
                              device=runner.device,
                              requires_grad=requires_grad)
        labels = None
        if label_dim > 0:
            rnd_labels = torch.randint(
                0, label_dim, (batch_size,), device=runner.device)
            labels = F.one_hot(rnd_labels, num_classes=label_dim)

        # Forward generator.
        G = runner.ddp_models['generator']
        G_kwargs = runner.model_kwargs_train['generator']
        with ddp_sync(G, sync=sync):
            return G(latents, labels, **G_kwargs)

    @staticmethod
    def run_D(runner, images, labels, sync=True):
        """Forwards discriminator."""
        # Augment the images.
        images = runner.augment(images, **runner.augment_kwargs)

        # Forward discriminator.
        D = runner.ddp_models['discriminator']
        D_kwargs = runner.model_kwargs_train['discriminator']
        with ddp_sync(D, sync=sync):
            return D(images, labels, **D_kwargs)

    @staticmethod
    def compute_grad_penalty(images, scores):
        """Computes gradient penalty."""
        with conv2d_gradfix.no_weight_gradients():
            image_grad = torch.autograd.grad(
                outputs=[scores.sum()],
                inputs=[images],
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
        grad_penalty = image_grad.square().sum((1, 2, 3))
        return grad_penalty

    def compute_pl_penalty(self, images, latents):
        """Computes perceptual path length penalty."""
        res_h, res_w = images.shape[2:4]
        pl_noise = torch.randn_like(images) / np.sqrt(res_h * res_w)
        with conv2d_gradfix.no_weight_gradients():
            code_grad = torch.autograd.grad(
                outputs=[(images * pl_noise).sum()],
                inputs=[latents],
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
        pl_length = code_grad.square().sum(2).mean(1).sqrt()
        pl_mean = self.pl_mean.lerp(pl_length.mean(), self.pl_decay)
        self.pl_mean.copy_(pl_mean.detach())
        pl_penalty = (pl_length - pl_mean).square()
        return pl_penalty

    def g_loss(self, runner, _data, sync=True):
        """Computes loss for generator."""
        fake_results = self.run_G(runner, sync=sync)
        fake_scores = self.run_D(runner,
                                 images=fake_results['image'],
                                 labels=fake_results['label'],
                                 sync=False)['score']
        g_loss = F.softplus(-fake_scores)
        runner.running_stats.update({'Loss/G': g_loss})

        return g_loss.mean()

    def g_reg(self, runner, _data, sync=True):
        """Computes the regularization loss for generator."""
        if runner.iter % self.pl_interval != 1 or self.pl_weight == 0.0:
            return None

        batch_size = max(runner.batch_size // self.pl_batch_shrink, 1)
        fake_results = self.run_G(runner,
                                  batch_size=batch_size,
                                  sync=sync,
                                  requires_grad=True)
        pl_penalty = self.compute_pl_penalty(images=fake_results['image'],
                                             latents=fake_results['wp'])
        runner.running_stats.update({'Loss/Path Length Penalty': pl_penalty})
        pl_penalty = pl_penalty * self.pl_weight * self.pl_interval

        return (fake_results['image'][:, 0, 0, 0] * 0 + pl_penalty).mean()

    def d_fake_loss(self, runner, _data, sync=True):
        """Computes discriminator loss on generated images."""
        fake_results = self.run_G(runner, sync=False)
        fake_scores = self.run_D(runner,
                                 images=fake_results['image'],
                                 labels=fake_results['label'],
                                 sync=sync)['score']
        d_fake_loss = F.softplus(fake_scores)
        runner.running_stats.update({'Loss/D Fake': d_fake_loss})

        return d_fake_loss.mean()

    def d_real_loss(self, runner, data, sync=True):
        """Computes discriminator loss on real images."""
        real_images = data['image'].detach()
        real_labels = data.get('label', None)
        real_scores = self.run_D(runner,
                                 images=real_images,
                                 labels=real_labels,
                                 sync=sync)['score']
        d_real_loss = F.softplus(-real_scores)
        runner.running_stats.update({'Loss/D Real': d_real_loss})

        # Adjust the augmentation strength if needed.
        if hasattr(runner.augment, 'prob_tracker'):
            runner.augment.prob_tracker.update(real_scores.sign())

        return d_real_loss.mean()

    def d_reg(self, runner, data, sync=True):
        """Computes the regularization loss for discriminator."""
        if runner.iter % self.r1_interval != 1 or self.r1_gamma == 0.0:
            return None

        real_images = data['image'].detach().requires_grad_(True)
        real_labels = data.get('label', None)
        real_scores = self.run_D(runner,
                                 images=real_images,
                                 labels=real_labels,
                                 sync=sync)['score']
        r1_penalty = self.compute_grad_penalty(images=real_images,
                                               scores=real_scores)
        runner.running_stats.update({'Loss/Real Gradient Penalty': r1_penalty})
        r1_penalty = r1_penalty * (self.r1_gamma * 0.5) * self.r1_interval

        # Adjust the augmentation strength if needed.
        if hasattr(runner.augment, 'prob_tracker'):
            runner.augment.prob_tracker.update(real_scores.sign())

        return (real_scores * 0 + r1_penalty).mean()
