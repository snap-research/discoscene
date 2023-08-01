"""Contains the runner for DiscoScene."""

from copy import deepcopy

from .base_runner import BaseRunner

import torch
__all__ = ['DiscoSceneRunner']


class DiscoSceneRunner(BaseRunner):
    """Defines the runner for DiscoScene."""

    def __init__(self, config):
        super().__init__(config)
        self.lod = getattr(self, 'lod', None)
        self.r1_gamma = getattr(self, 'r1_gamma', None)
        self.D_repeats = self.config.get('D_repeats', 1)
        #TODO comment this in future
        self.grad_clip = self.config.get('grad_clip', None)
        self.add_scene_d = self.config.add_scene_d
        
        if self.grad_clip is not None:
            self.running_stats.add(
                f'g_grad_norm', log_format='.3f', log_strategy='AVERAGE')
            self.running_stats.add(
                f'd_grad_norm', log_format='.3f', log_strategy='AVERAGE')
            if self.add_scene_d:
                self.running_stats.add(
                    f'object_d_grad_norm', log_format='.3f', log_strategy='AVERAGE')

    def build_models(self): 
        super().build_models() 
        self.g_ema_img = self.config.models['generator'].get( 'g_ema_img', 10000) 
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

    def clip_grads(self, params, is_g=False, named_params=None):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            if is_g:
                params_ = list(filter(lambda p: p[1].requires_grad and p[1].grad is not None, named_params))
                print(len(params_))
                print([(name, param.grad.mean().item()) for name, param in params_])
            return torch.nn.utils.clip_grad_norm_(params, **self.grad_clip)
        else:
            self.logger.info('There exsits no parameters to clip!')
            raise NotImplementedError

    def train_step(self, data, **train_kwargs):
        # print(f'add_scene_d:{self.add_scene_d}')
        if self.amp_scaler.get_scale() < 1:
            self.amp_scaler.update(1.)

        # Set level-of-details.
        G = self.models['generator']
        D = self.models['discriminator']
        Gs = self.models['generator_smooth']

        if self.lod is None: self.lod = 0
        G.synthesis.lod.data.fill_(self.lod)
        if hasattr(G, 'bg_synthesis'):
            G.bg_synthesis.lod.data.fill_(self.lod)
        if hasattr(G, 'superresolution'):
            G.superresolution.lod.data.fill_(self.lod)
        D.lod.data.fill_(self.lod)
        Gs.synthesis.lod.data.fill_(self.lod)
        if hasattr(Gs, 'bg_synthesis'):
            Gs.bg_synthesis.lod.data.fill_(self.lod)
        if hasattr(Gs, 'superresolution'):
            Gs.superresolution.lod.data.fill_(self.lod)
        if self.config.object_use_pg:
            self.models['discriminator_object'].lod.data.fill_(self.lod)

        # Update discriminator.
        self.models['discriminator'].requires_grad_(True)
        self.models['generator'].requires_grad_(False)
        if self.add_scene_d: 
            self.models['discriminator_object'].requires_grad_(True)

       
        d_loss = self.loss.d_loss(self, data, sync=True, update_kwargs=dict(r1_gamma=self.r1_gamma))

        # TODO set_to_none
        set_to_none = None 
        self.zero_grad_optimizer('discriminator', set_to_none)
        if self.add_scene_d: 
            self.zero_grad_optimizer('discriminator_object', set_to_none)

        d_loss.backward()

        self.unscale_optimizer('discriminator')
        if self.add_scene_d: 
            self.unscale_optimizer('discriminator_object')

        if self.grad_clip is not None:
            d_grad_norm = self.clip_grads(self.models['discriminator'].parameters())
            if d_grad_norm is not None:
                self.running_stats.update({'d_grad_norm': d_grad_norm.item()})
            if self.add_scene_d:
                d_object_grad_norm = self.clip_grads(self.models['discriminator_object'].parameters())
                if d_object_grad_norm is not None:
                    self.running_stats.update({'object_d_grad_norm': d_object_grad_norm.item()})

        self.step_optimizer('discriminator')
        if self.add_scene_d: 
            self.step_optimizer('discriminator_object')

        # Life-long update for generator.
        beta = 0.5 ** (self.minibatch / self.g_ema_img)
        self.running_stats.update({'Misc/Gs Beta': beta})
        self.smooth_model(src=self.models['generator'],
                          avg=self.models['generator_smooth'],
                          beta=beta)

        # Update generator.
        if self.iter % self.D_repeats == 0:
            self.models['discriminator'].requires_grad_(False)
            if self.add_scene_d: 
                self.models['discriminator_object'].requires_grad_(False)
            self.models['generator'].requires_grad_(True)

            g_loss = self.loss.g_loss(self, data, sync=True, update_kwargs=dict(r1_gamma=self.r1_gamma))
            self.zero_grad_optimizer('generator', set_to_none)
            g_loss.backward()
            self.unscale_optimizer('generator')
            # import ipdb;ipdb.set_trace()
            if self.grad_clip is not None:
                g_grad_norm = self.clip_grads(self.models['generator'].parameters(), is_g=False, named_params=self.models['generator'].named_parameters())
                if g_grad_norm is not None:
                    self.running_stats.update({'g_grad_norm': g_grad_norm.item()})
            self.step_optimizer('generator')

        # Update automatic mixed-precision scaler.
        self.amp_scaler.update()

