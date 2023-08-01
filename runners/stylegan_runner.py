# python3.7
"""Contains the runner for StyleGAN."""

from copy import deepcopy

from .base_runner import BaseRunner

__all__ = ['StyleGANRunner']


class StyleGANRunner(BaseRunner):
    """Defines the runner for StyleGAN."""

    def __init__(self, config):
        super().__init__(config)
        self.lod = getattr(self, 'lod', 0.0)
        self.D_repeats = self.config.get('D_repeats', 1)

    def build_models(self):
        super().build_models()
        self.g_ema_img = self.config.models['generator'].get(
            'g_ema_img', 10_000)
        if 'generator_smooth' not in self.models:
            self.models['generator_smooth'] = deepcopy(self.models['generator'])
            self.model_kwargs_init['generator_smooth'] = deepcopy(
                self.model_kwargs_init['generator'])
        if 'generator_smooth' not in self.model_kwargs_val:
            self.model_kwargs_val['generator_smooth'] = deepcopy(
                self.model_kwargs_val['generator'])

    def build_loss(self):
        super().build_loss()
        self.running_stats.add('Misc/Gs Beta',
                               log_name='Gs_beta',
                               log_format='.4f',
                               log_strategy='CURRENT')

    def train_step(self, data):
        # Set level-of-details.
        G = self.models['generator']
        D = self.models['discriminator']
        Gs = self.models['generator_smooth']
        G.synthesis.lod.data.fill_(self.lod)
        D.lod.data.fill_(self.lod)
        Gs.synthesis.lod.data.fill_(self.lod)

        # Update discriminator.
        self.models['discriminator'].requires_grad_(True)
        self.models['generator'].requires_grad_(False)
        d_loss = self.loss.d_loss(self, data, sync=True)
        self.zero_grad_optimizer('discriminator')
        d_loss.backward()
        self.step_optimizer('discriminator')

        # Life-long update for generator.
        beta = 0.5 ** (self.minibatch / self.g_ema_img)
        self.running_stats.update({'Misc/Gs Beta': beta})
        self.smooth_model(src=self.models['generator'],
                          avg=self.models['generator_smooth'],
                          beta=beta)

        # Update generator.
        if self.iter % self.D_repeats == 0:
            self.models['discriminator'].requires_grad_(False)
            self.models['generator'].requires_grad_(True)
            g_loss = self.loss.g_loss(self, data, sync=True)
            self.zero_grad_optimizer('generator')
            g_loss.backward()
            self.step_optimizer('generator')

        # Update automatic mixed-precision scaler.
        self.amp_scaler.update()
