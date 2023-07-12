# python3.7
"""Converts StyleGAN2 model weights.

The models can be trained through OR released by the repository:

https://github.com/NVlabs/stylegan2
"""

import os
import sys
from copy import deepcopy
import pickle
import numpy as np

import torch

from utils.visualizers import HtmlVisualizer
from utils.image_utils import postprocess_image
from utils.tf_utils import import_tf
from .base_converter import BaseConverter

tf = import_tf()

__all__ = ['StyleGAN2Converter']

OFFICIAL_CODE_DIR = 'stylegan2_official'
BASE_DIR = os.path.dirname(os.path.relpath(__file__))
CODE_PATH = os.path.join(BASE_DIR, OFFICIAL_CODE_DIR)

TRUNC_PSI = 0.5
TRUNC_LAYERS = 8
FUSED_MODULATE = False

IMPL = 'cuda'


class StyleGAN2Converter(BaseConverter):
    """Defines the converter for converting StyleGAN2 model."""

    def __init__(self, verbose_log=False):
        if tf is None:
            raise SystemExit('Converting StyleGAN2 relies on the module '
                             '`tensorflow`, which is not installed in the '
                             'current environment!')

        super().__init__(verbose_log=verbose_log)
        self.sess = tf.compat.v1.InteractiveSession()
        self.source_executable = tf.test.is_built_with_cuda()

    def __del__(self):
        if tf is not None:
            self.sess.close()

    def load_source(self, path):
        sys.path.insert(0, CODE_PATH)
        from dnnlib import tflib  # pylint: disable=import-error, import-outside-toplevel
        self.tflib = tflib  # pylint: disable=attribute-defined-outside-init
        with open(path, 'rb') as f:
            G, D, Gs = pickle.load(f)
        sys.path.pop(0)
        self.src_models['generator'] = G
        self.src_models['discriminator'] = D
        self.src_models['generator_smooth'] = Gs

    def parse_model_config(self):
        G = self.src_models['generator']
        z_dim = G.input_shapes[0][1]
        label_dim = G.input_shapes[1][1]
        w_dim = G.components.mapping.output_shape[2]
        image_channels = G.output_shape[1]
        resolution = G.output_shape[2]

        G_kwargs = self.src_models['generator'].static_kwargs
        G_normalize_z = (label_dim == 0)
        G_mapping_layers = G_kwargs.get('mapping_layers', 8)
        G_embedding_dim = G_kwargs.get('label_fmaps', z_dim)
        G_embedding_bias = False
        G_embedding_use_wscale = False
        G_normalize_embedding = False
        G_normalize_embedding_latent = True
        G_arch = G_kwargs.get('architecture', 'skip')
        G_arch = 'origin' if G_arch == 'orig' else G_arch
        G_fmaps_base = G_kwargs.get('fmap_base', 16 << 10) * 2
        G_fmaps_max = G_kwargs.get('fmap_max', 512)
        G_conv_clamp = None

        D_kwargs = self.src_models['discriminator'].static_kwargs
        D_arch = D_kwargs.get('architecture', 'resnet')
        D_arch = 'origin' if D_arch == 'orig' else D_arch
        D_fmaps_base = D_kwargs.get('fmap_base', 16 << 10) * 2
        D_fmaps_max = D_kwargs.get('fmap_max', 512)
        D_conv_clamp = None
        D_embedding_dim = 0
        D_embedding_bias = False
        D_embedding_use_wscale = False
        D_normalize_embedding = False
        D_mapping_layers = 0

        self.dst_kwargs['generator'] = dict(
            model_type='StyleGAN2Generator',
            resolution=resolution,
            z_dim=z_dim,
            w_dim=w_dim,
            normalize_z=G_normalize_z,
            mapping_layers=G_mapping_layers,
            embedding_dim=G_embedding_dim,
            embedding_bias=G_embedding_bias,
            embedding_use_wscale=G_embedding_use_wscale,
            normalize_embedding=G_normalize_embedding,
            normalize_embedding_latent=G_normalize_embedding_latent,
            label_dim=label_dim,
            image_channels=image_channels,
            architecture=G_arch,
            fmaps_base=G_fmaps_base,
            fmaps_max=G_fmaps_max,
            conv_clamp=G_conv_clamp)
        self.dst_kwargs['discriminator'] = dict(
            model_type='StyleGAN2Discriminator',
            resolution=resolution,
            label_dim=label_dim,
            image_channels=image_channels,
            architecture=D_arch,
            fmaps_base=D_fmaps_base,
            fmaps_max=D_fmaps_max,
            conv_clamp=D_conv_clamp,
            embedding_dim=D_embedding_dim,
            embedding_bias=D_embedding_bias,
            embedding_use_wscale=D_embedding_use_wscale,
            normalize_embedding=D_normalize_embedding,
            mapping_layers=D_mapping_layers)
        self.dst_kwargs['generator_smooth'] = deepcopy(
            self.dst_kwargs['generator'])

    @staticmethod
    def convert_generator(src_model, dst_model, log_fn=None):
        """Converts the generator weights."""
        # Get source weights.
        src_vars = dict(src_model.__getstate__()['variables'])
        src_vars.update(
            dict(src_model.components.mapping.__getstate__()['variables']))
        src_vars.update(
            dict(src_model.components.synthesis.__getstate__()['variables']))
        # Get target weights.
        dst_state = deepcopy(dst_model.state_dict())
        # Get variable mapping.
        dst_to_src_mapping = dst_model.pth_to_tf_var_mapping
        # Convert.
        for dst_name, src_name in dst_to_src_mapping.items():
            src_name = src_name.replace('LabelEmbed', 'LabelConcat')
            assert src_name in src_vars, f'Var `{src_name}` missing.'
            assert dst_name in dst_state, f'Var `{dst_name}` missing.'
            if log_fn is not None:
                log_fn(f'Converting `{src_name}` to `{dst_name}`.',
                       indent_level=2, is_verbose=True)
            var = torch.from_numpy(np.array(src_vars[src_name]))
            if 'weight' in src_name:
                if 'Dense' in src_name:
                    var = var.permute(1, 0)
                elif 'mod_weight' in src_name:
                    var = var.permute(1, 0)
                elif 'LabelConcat' in src_name:
                    var = var.permute(1, 0)
                elif 'Conv0_up/weight' in src_name:
                    var = var.permute(3, 2, 0, 1).flip(2, 3)
                elif 'Skip' in src_name:
                    var = var.permute(3, 2, 0, 1).flip(2, 3)
                else:
                    var = var.permute(3, 2, 0, 1)
            if 'bias' in src_name:
                if 'mod_bias' in src_name:
                    var = var + 1
            dst_state[dst_name] = var
        return dst_state

    @staticmethod
    def convert_discriminator(src_model, dst_model, log_fn=None):
        """Converts the discriminator weights."""
        # Get source weights.
        src_vars = dict(src_model.__getstate__()['variables'])
        # Get target weights.
        dst_state = deepcopy(dst_model.state_dict())
        # Get variable mapping.
        dst_to_src_mapping = dst_model.pth_to_tf_var_mapping
        # Convert.
        for dst_name, src_name in dst_to_src_mapping.items():
            assert src_name in src_vars, f'Var `{src_name}` missing.'
            assert dst_name in dst_state, f'Var `{dst_name}` missing.'
            if log_fn is not None:
                log_fn(f'Converting `{src_name}` to `{dst_name}`.',
                       indent_level=2, is_verbose=True)
            var = torch.from_numpy(np.array(src_vars[src_name]))
            if 'weight' in src_name:
                if 'Dense' in src_name:
                    var = var.permute(1, 0)
                elif 'Output' in src_name:
                    var = var.permute(1, 0)
                else:
                    var = var.permute(3, 2, 0, 1)
            dst_state[dst_name] = var
        return dst_state

    def convert(self):
        self.parse_model_config()
        self.build_target()
        for model_name, src_model in self.src_models.items():
            dst_model = self.dst_models[model_name]
            if model_name in ['generator', 'generator_smooth']:
                convert_fn = self.convert_generator
            elif model_name in ['discriminator']:
                convert_fn = self.convert_discriminator
            self.logger.print(f'Converting `{model_name}` ...',
                              indent_level=1, is_verbose=True)
            self.dst_states[model_name] = convert_fn(
                src_model, dst_model, log_fn=self.logger.print)

    def test_forward(self, num, save_test_image=False):
        assert num > 0

        if save_test_image:
            html = HtmlVisualizer(num_rows=num, num_cols=3)
            html.set_headers(['Index', 'Before Conversion', 'After Conversion'])
            for i in range(num):
                html.set_cell(i, 0, text=f'{i}')

        G_src = self.src_models['generator']
        D_src = self.src_models['discriminator']
        Gs_src = self.src_models['generator_smooth']
        G_dst = self.dst_models['generator']
        D_dst = self.dst_models['discriminator']
        Gs_dst = self.dst_models['generator_smooth']
        G_dst.load_state_dict(self.dst_states['generator'])
        D_dst.load_state_dict(self.dst_states['discriminator'])
        Gs_dst.load_state_dict(self.dst_states['generator_smooth'])
        G_dst.eval().cuda()
        D_dst.eval().cuda()
        Gs_dst.eval().cuda()
        latent_dim = G_dst.latent_dim
        label_dim = G_dst.label_dim

        gs_error = 0.0  # The error of Gs(z) between source and target.
        gs_mean = 0.0  # The mean of Gs(z) from source.
        dg_error = 0.0  # The error of D(G(z)) between source and target.
        dg_mean = 0.0  # The mean of D(G(z)) from source.
        for i in range(num):
            ##### Test Gs(z) #####
            # Latent code.
            latent = np.random.randn(1, *latent_dim)
            latent_pth = torch.from_numpy(latent).type(torch.FloatTensor).cuda()
            # Label.
            label = np.zeros((1, label_dim), np.float32)
            if label_dim:
                label_id = np.random.randint(label_dim)
                label[0, label_id] = 1.0
                label_pth = torch.from_numpy(label).to(latent_pth)
            else:
                label_pth = None
            # Forward source.
            src_image = Gs_src.run(latent, label,
                                   truncation_psi=TRUNC_PSI,
                                   truncation_cutoff=TRUNC_LAYERS,
                                   fused_modconv=FUSED_MODULATE,
                                   randomize_noise=False)
            # Forward target.
            with torch.no_grad():
                dst_image = Gs_dst(latent_pth, label_pth,
                                   trunc_psi=TRUNC_PSI,
                                   trunc_layers=TRUNC_LAYERS,
                                   noise_mode='const',
                                   fused_modulate=FUSED_MODULATE,
                                   impl=IMPL)['image']
            # Compare.
            error = self.mean_error(src_image, dst_image)
            mean = self.mean_error(src_image, None)
            self.logger.print(f'Test Gs(z) {i:03d}: '
                              f'Error {error:.3e} '
                              f'(source mean {mean:.3e}).',
                              indent_level=1, is_verbose=True)
            gs_error += error
            gs_mean += mean

            if save_test_image:
                html.set_cell(i, 1, image=postprocess_image(src_image)[0])
                dst_image = dst_image.detach().cpu().numpy()
                html.set_cell(i, 2, image=postprocess_image(dst_image)[0])

            ##### Test D(G(z)) #####
            # Latent code.
            latent = np.random.randn(1, *latent_dim)
            latent_pth = torch.from_numpy(latent).type(torch.FloatTensor).cuda()
            # Label.
            label = np.zeros((1, label_dim), np.float32)
            if label_dim:
                label_id = np.random.randint(label_dim)
                label[0, label_id] = 1.0
                label_pth = torch.from_numpy(label).to(latent_pth)
            else:
                label_pth = None
            # Forward source.
            src_image = G_src.run(latent, label,
                                  truncation_psi=TRUNC_PSI,
                                  truncation_cutoff=TRUNC_LAYERS,
                                  randomize_noise=False,
                                  fused_modconv=FUSED_MODULATE)
            src_score = D_src.run(src_image, label)
            # Forward target.
            with torch.no_grad():
                dst_image = G_dst(latent_pth, label_pth,
                                  trunc_psi=TRUNC_PSI,
                                  trunc_layers=TRUNC_LAYERS,
                                  noise_mode='const',
                                  fused_modulate=FUSED_MODULATE,
                                  impl=IMPL)['image']
                dst_score = D_dst(dst_image, label_pth, impl=IMPL)['score']
            # Compare.
            error = self.mean_error(src_score, dst_score)
            mean = self.mean_error(src_score, None)
            self.logger.print(f'Test D(G(z)) {i:03d}: '
                              f'Error {error:.3e} '
                              f'(source mean {mean:.3e}).',
                              indent_level=1, is_verbose=True)
            dg_error += error
            dg_mean += mean

        self.logger.print(f'Tested Gs(z): '
                          f'Error {gs_error / num:.3e} '
                          f'(source mean {gs_mean / num:.3e}).')
        self.logger.print(f'Tested D(G(z)): '
                          f'Error {dg_error / num:.3e} '
                          f'(source mean {dg_mean / num:.3e}).')

        if save_test_image:
            html.save(f'{self.dst_path}.conversion_forward_test.html')

    def test_backward(self, num, learning_rate=0.01):
        assert num > 0

        G_src = self.src_models['generator']
        D_src = self.src_models['discriminator']
        G_dst = self.dst_models['generator']
        D_dst = self.dst_models['discriminator']
        G_dst.load_state_dict(self.dst_states['generator'])
        D_dst.load_state_dict(self.dst_states['discriminator'])
        G_dst.train().cuda()
        D_dst.train().cuda()
        latent_dim = G_dst.latent_dim
        label_dim = G_dst.label_dim

        # Build graph for source model.
        with tf.name_scope('Inputs'), tf.device('/cpu:0'):
            lrate_in = tf.placeholder(tf.float32, name='lrate_in', shape=[])
            minibatch_in = tf.placeholder(tf.int32, name='minibatch_in',
                                          shape=[])
            latent_in = tf.placeholder(tf.float32, name='latents_in',
                                       shape=[1, *latent_dim])
            label_in = tf.placeholder(tf.float32, name='labels_in',
                                      shape=[1, label_dim])
        feed_dict = {lrate_in: learning_rate, minibatch_in: 1}

        G_src_opt = self.tflib.Optimizer(
            name='TrainG',
            learning_rate=lrate_in,
            tf_optimizer='tf.train.MomentumOptimizer',
            momentum=0.0)
        D_src_opt = self.tflib.Optimizer(
            name='TrainD',
            learning_rate=lrate_in,
            tf_optimizer='tf.train.MomentumOptimizer',
            momentum=0.0)

        src_image = G_src.get_output_for(latent_in, label_in,
                                         dlatent_avg_beta=None,
                                         style_mixing_prob=None,
                                         randomize_noise=False,
                                         fused_modconv=FUSED_MODULATE,
                                         is_training=True)
        src_score = D_src.get_output_for(src_image, label_in, is_training=True)
        G_src_loss = -tf.nn.softplus(src_score)
        D_src_loss = tf.nn.softplus(src_score)
        G_src_opt.register_gradients(
            tf.reduce_mean(G_src_loss), G_src.trainables)
        D_src_opt.register_gradients(
            tf.reduce_mean(D_src_loss), D_src.trainables)
        G_src_train_op = G_src_opt.apply_updates()
        D_src_train_op = D_src_opt.apply_updates()

        # Build optimizer for target model.
        G_dst_opt = torch.optim.SGD(G_dst.parameters(),
                                    lr=learning_rate,
                                    momentum=0.0,
                                    weight_decay=0.0)
        D_dst_opt = torch.optim.SGD(D_dst.parameters(),
                                    lr=learning_rate,
                                    momentum=0.0,
                                    weight_decay=0.0)

        self.logger.print('Before training ...', indent_level=1)
        self.logger.print('Checking generator ...', indent_level=2)
        with torch.no_grad():
            temp_model = deepcopy(G_dst).eval().cpu()
            temp_src_state = self.convert_generator(G_src, temp_model)
            temp_dst_state = temp_model.state_dict()
            self.check_weight(temp_src_state, temp_dst_state)
        self.logger.print('Checking discriminator ...', indent_level=2)
        with torch.no_grad():
            temp_model = deepcopy(D_dst).eval().cpu()
            temp_src_state = self.convert_discriminator(D_src, temp_model)
            temp_dst_state = temp_model.state_dict()
            self.check_weight(temp_src_state, temp_dst_state)

        for i in range(num):
            ##### Train discriminator #####
            # Latent code.
            latent = np.random.randn(1, *latent_dim)
            latent_pth = torch.from_numpy(latent).type(torch.FloatTensor).cuda()
            # Label.
            label = np.zeros((1, label_dim), np.float32)
            if label_dim:
                label_id = np.random.randint(label_dim)
                label[0, label_id] = 1.0
                label_pth = torch.from_numpy(label).to(latent_pth)
            else:
                label_pth = None
            feed_dict[latent_in] = latent
            feed_dict[label_in] = label
            # Train source.
            _, src_loss, src_score_val = self.tflib.run(
                [D_src_train_op, D_src_loss, src_score], feed_dict)
            # Train target.
            for param in G_dst.parameters():
                param.requires_grad = False
            for param in D_dst.parameters():
                param.requires_grad = True
            dst_image = G_dst(latent_pth, label_pth,
                              w_moving_decay=1.0,
                              style_mixing_prob=0,
                              noise_mode='const',
                              fused_modulate=FUSED_MODULATE,
                              impl=IMPL)['image']
            dst_score = D_dst(dst_image, label_pth, impl=IMPL)['score']
            D_dst_loss = torch.nn.functional.softplus(dst_score).mean()
            D_dst_opt.zero_grad()
            D_dst_loss.backward()
            D_dst_opt.step()
            # Compare.
            self.logger.print(f'Step {i:03d}, train discriminator ... ('
                              f'source score {src_score_val[0][0]:.6e}, '
                              f'target score {dst_score[0][0].item():.6e}, '
                              f'source loss {src_loss[0][0]:.6e}, '
                              f'target loss {D_dst_loss.item():.6e})',
                              indent_level=1)
            self.logger.print('Checking generator ...', indent_level=2)
            with torch.no_grad():
                temp_model = deepcopy(G_dst).eval().cpu()
                temp_src_state = self.convert_generator(G_src, temp_model)
                temp_dst_state = temp_model.state_dict()
                init_state = self.dst_states['generator']
                self.check_weight(temp_src_state, temp_dst_state, init_state)
            self.logger.print('Checking discriminator ...', indent_level=2)
            with torch.no_grad():
                temp_model = deepcopy(D_dst).eval().cpu()
                temp_src_state = self.convert_discriminator(D_src, temp_model)
                temp_dst_state = temp_model.state_dict()
                init_state = self.dst_states['discriminator']
                self.check_weight(temp_src_state, temp_dst_state, init_state)

            ##### Train generator #####
            # Latent code.
            latent = np.random.randn(1, *latent_dim)
            latent_pth = torch.from_numpy(latent).type(torch.FloatTensor).cuda()
            # Label.
            label = np.zeros((1, label_dim), np.float32)
            if label_dim:
                label_id = np.random.randint(label_dim)
                label[0, label_id] = 1.0
                label_pth = torch.from_numpy(label).to(latent_pth)
            else:
                label_pth = None
            feed_dict[latent_in] = latent
            feed_dict[label_in] = label
            # Train source.
            _, src_loss, src_score_val = self.tflib.run(
                [G_src_train_op, G_src_loss, src_score], feed_dict)
            # Train target.
            for param in G_dst.parameters():
                param.requires_grad = True
            for param in D_dst.parameters():
                param.requires_grad = False
            dst_image = G_dst(latent_pth, label_pth,
                              w_moving_decay=1.0,
                              style_mixing_prob=0,
                              noise_mode='const',
                              fused_modulate=FUSED_MODULATE,
                              impl=IMPL)['image']
            dst_score = D_dst(dst_image, label_pth, impl=IMPL)['score']
            G_dst_loss = -torch.nn.functional.softplus(dst_score).mean()
            G_dst_opt.zero_grad()
            G_dst_loss.backward()
            G_dst_opt.step()
            # Compare.
            self.logger.print(f'Step {i:03d}, train generator ... ('
                              f'source score {src_score_val[0][0]:.6e}, '
                              f'target score {dst_score[0][0].item():.6e}, '
                              f'source loss {src_loss[0][0]:.6e}, '
                              f'target loss {G_dst_loss.item():.6e})',
                              indent_level=1)
            self.logger.print('Checking generator ...', indent_level=2)
            with torch.no_grad():
                temp_model = deepcopy(G_dst).eval().cpu()
                temp_src_state = self.convert_generator(G_src, temp_model)
                temp_dst_state = temp_model.state_dict()
                init_state = self.dst_states['generator']
                self.check_weight(temp_src_state, temp_dst_state, init_state)
            self.logger.print('Checking discriminator ...', indent_level=2)
            with torch.no_grad():
                temp_model = deepcopy(D_dst).eval().cpu()
                temp_src_state = self.convert_discriminator(D_src, temp_model)
                temp_dst_state = temp_model.state_dict()
                init_state = self.dst_states['discriminator']
                self.check_weight(temp_src_state, temp_dst_state, init_state)
