# python3.7
"""Defines loss functions for GAN training."""

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from utils.dist_utils import ddp_sync
from .base_loss import BaseLoss

__all__ = ['StyleGANLoss']


class StyleGANLoss(BaseLoss):
    """Contains the class to compute losses for training StyleGAN.

    Basically, this class contains the computation of adversarial loss for both
    generator and discriminator, as well as gradient penalty as the
    regularization for discriminator.
    """

    def __init__(self, runner, d_loss_kwargs=None, g_loss_kwargs=None):
        """Initializes with models and arguments for computing losses."""

        # Settings for losses.
        self.d_loss_kwargs = d_loss_kwargs or dict()
        self.g_loss_kwargs = g_loss_kwargs or dict()
        self.r1_gamma = self.d_loss_kwargs.get('r1_gamma', 10.0)
        self.r2_gamma = self.d_loss_kwargs.get('r2_gamma', 0.0)

        runner.running_stats.add('Loss/D Real',
                                 log_name='loss_d_real',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')
        runner.running_stats.add('Loss/D Fake',
                                 log_name='loss_d_fake',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')
        runner.running_stats.add('Loss/G',
                                 log_name='loss_g',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')
        if self.r1_gamma > 0.0:
            runner.running_stats.add('Loss/Real Grad Penalty',
                                     log_name='loss_gp_real',
                                     log_format='.2e',
                                     log_strategy='AVERAGE')
        if self.r2_gamma > 0.0:
            runner.running_stats.add('Loss/Fake Grad Penalty',
                                     log_name='loss_gp_fake',
                                     log_format='.2e',
                                     log_strategy='AVERAGE')

        # Log loss settings.
        runner.logger.info('real gradient penalty:', indent_level=1)
        runner.logger.info(f'r1_gamma: {self.r1_gamma}', indent_level=2)
        runner.logger.info('fake gradient penalty:', indent_level=1)
        runner.logger.info(f'r2_gamma: {self.r2_gamma}', indent_level=2)

    @staticmethod
    def preprocess_image(images, lod=0):
        """Pre-process images to support progressive training."""
        # Downsample to the resolution of the current phase (level-of-details).
        for _ in range(int(lod)):
            images = F.avg_pool2d(
                images, kernel_size=2, stride=2, padding=0)
        # Transition from the previous phase (level-of-details) if needed.
        if lod != int(lod):
            downsampled_images = F.avg_pool2d(
                images, kernel_size=2, stride=2, padding=0)
            upsampled_images = F.interpolate(
                downsampled_images, scale_factor=2, mode='nearest')
            alpha = lod - int(lod)
            images = images * (1 - alpha) + upsampled_images * alpha
        # Upsample back to the resolution of the model.
        if int(lod) == 0:
            return images
        return F.interpolate(
            images, scale_factor=(2 ** int(lod)), mode='nearest')

    @staticmethod
    def run_G(runner, latent_requires_grad=False, sync=True):
        """Forwards generator."""
        # Prepare latents and labels.
        batch_size = runner.batch_size
        latent_dim = runner.models['generator'].latent_dim
        label_dim = runner.models['generator'].label_dim
        latents = torch.randn((batch_size, *latent_dim), device=runner.device)
        latents.requires_grad_(latent_requires_grad)
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
    def compute_grad_penalty(images, scores, amp_scaler):
        """Computes gradient penalty."""
        # Scales the scores for autograd.grad's backward pass.
        # If disable amp, the scaler will always be 1.
        scores = amp_scaler.scale(scores)

        image_grad = torch.autograd.grad(
            outputs=[scores.sum()],
            inputs=[images],
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        if amp_scaler.is_enabled():
            image_grad = image_grad / amp_scaler.get_scale()

        with autocast(enabled=amp_scaler.is_enabled()):
            penalty = image_grad.square().sum((1, 2, 3))

        return penalty

    def d_loss(self, runner, data, sync=True):
        """Computes loss for discriminator."""
        # Train with real samples.
        real_images = self.preprocess_image(
            data['image'], lod=runner.lod).detach()
        real_images.requires_grad_(self.r1_gamma > 0.0)
        real_labels = data.get('label', None)
        real_scores = self.run_D(runner,
                                 images=real_images,
                                 labels=real_labels,
                                 sync=sync)['score']
        with autocast(enabled=runner.enable_amp):
            d_real_loss = F.softplus(-real_scores)
            runner.running_stats.update({'Loss/D Real': d_real_loss})
            d_real_loss = runner.amp_scaler.scale(d_real_loss)

        # Adjust the augmentation strength if needed.
        if hasattr(runner.augment, 'prob_tracker'):
            runner.augment.prob_tracker.update(real_scores.sign())

        # Train with fake samples.
        fake_results = self.run_G(
            runner, latent_requires_grad=(self.r2_gamma > 0.0), sync=False)
        fake_scores = self.run_D(runner,
                                 images=fake_results['image'],
                                 labels=fake_results['label'],
                                 sync=sync)['score']
        with autocast(enabled=runner.enable_amp):
            d_fake_loss = F.softplus(fake_scores)
            runner.running_stats.update({'Loss/D Fake': d_fake_loss})
            d_fake_loss = runner.amp_scaler.scale(d_fake_loss)

        # Gradient penalty with real samples.
        r1_penalty = torch.zeros_like(d_real_loss)
        if self.r1_gamma > 0.0:
            r1_penalty = self.compute_grad_penalty(
                images=real_images,
                scores=real_scores,
                amp_scaler=runner.amp_scaler)
            runner.running_stats.update({'Loss/Real Grad Penalty': r1_penalty})
            r1_penalty = runner.amp_scaler.scale(r1_penalty)

        # Gradient penalty with fake samples.
        r2_penalty = torch.zeros_like(d_fake_loss)
        if self.r2_gamma > 0.0:
            r2_penalty = self.compute_grad_penalty(
                images=fake_results['image'],
                scores=fake_scores,
                amp_scaler=runner.amp_scaler)
            runner.running_stats.update({'Loss/Fake Grad Penalty': r2_penalty})
            r2_penalty = runner.amp_scaler.scale(r2_penalty)

        return (d_real_loss +
                d_fake_loss +
                r1_penalty * (self.r1_gamma * 0.5) +
                r2_penalty * (self.r2_gamma * 0.5)).mean()

    def g_loss(self, runner, _data, sync=True):
        """Computes loss for generator."""
        fake_results = self.run_G(runner, sync=sync)
        fake_scores = self.run_D(runner,
                                 images=fake_results['image'],
                                 labels=fake_results['label'],
                                 sync=False)['score']
        with autocast(enabled=runner.enable_amp):
            g_loss = F.softplus(-fake_scores)
            runner.running_stats.update({'Loss/G': g_loss})
            g_loss = runner.amp_scaler.scale(g_loss)

        return g_loss.mean()
