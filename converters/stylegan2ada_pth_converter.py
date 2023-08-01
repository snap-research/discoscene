# python3.7
"""Converts StyleGAN2-ADA (PyTorch version) model weights.

The models can be trained through OR released by the repository:

https://github.com/NVlabs/stylegan2-ada-pytorch
"""

import os
import sys
import re
from copy import deepcopy
import pickle
import numpy as np

import torch

from utils.visualizers import HtmlVisualizer
from utils.image_utils import postprocess_image
from .base_converter import BaseConverter

__all__ = ['StyleGAN2ADAPTHConverter']

OFFICIAL_CODE_DIR = 'stylegan2ada_pth_official'
BASE_DIR = os.path.dirname(os.path.relpath(__file__))
CODE_PATH = os.path.join(BASE_DIR, OFFICIAL_CODE_DIR)

TRUNC_PSI = 0.5
TRUNC_LAYERS = 8
FUSED_MODULATE = False

FORCE_FP32 = False
NUM_FP16_RES = 4  # Default setting in StyleGAN2-ADA. (Be careful to modify)

IMPL = 'cuda'

# The following two dictionary of mapping patterns are modified from
# https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/legacy.py
G_PTH_TO_TF_VAR_MAPPING_PATTERN = {
    r'mapping\.w_avg':
        lambda:   'dlatent_avg',
    r'mapping\.embed\.weight':
        lambda:   'LabelEmbed/weight',
    r'mapping\.embed\.bias':
        lambda:   'LabelEmbed/bias',
    r'mapping\.fc(\d+)\.weight':
        lambda i: f'Dense{i}/weight',
    r'mapping\.fc(\d+)\.bias':
        lambda i: f'Dense{i}/bias',
    r'synthesis\.b4\.const':
        lambda:   '4x4/Const/const',
    r'synthesis\.b4\.conv1\.weight':
        lambda:   '4x4/Conv/weight',
    r'synthesis\.b4\.conv1\.bias':
        lambda:   '4x4/Conv/bias',
    r'synthesis\.b4\.conv1\.noise_const':
        lambda:   'noise0',
    r'synthesis\.b4\.conv1\.noise_strength':
        lambda:   '4x4/Conv/noise_strength',
    r'synthesis\.b4\.conv1\.affine\.weight':
        lambda:   '4x4/Conv/mod_weight',
    r'synthesis\.b4\.conv1\.affine\.bias':
        lambda:   '4x4/Conv/mod_bias',
    r'synthesis\.b(\d+)\.conv0\.weight':
        lambda r: f'{r}x{r}/Conv0_up/weight',
    r'synthesis\.b(\d+)\.conv0\.bias':
        lambda r: f'{r}x{r}/Conv0_up/bias',
    r'synthesis\.b(\d+)\.conv0\.noise_const':
        lambda r: f'noise{int(np.log2(int(r)))*2-5}',
    r'synthesis\.b(\d+)\.conv0\.noise_strength':
        lambda r: f'{r}x{r}/Conv0_up/noise_strength',
    r'synthesis\.b(\d+)\.conv0\.affine\.weight':
        lambda r: f'{r}x{r}/Conv0_up/mod_weight',
    r'synthesis\.b(\d+)\.conv0\.affine\.bias':
        lambda r: f'{r}x{r}/Conv0_up/mod_bias',
    r'synthesis\.b(\d+)\.conv1\.weight':
        lambda r: f'{r}x{r}/Conv1/weight',
    r'synthesis\.b(\d+)\.conv1\.bias':
        lambda r: f'{r}x{r}/Conv1/bias',
    r'synthesis\.b(\d+)\.conv1\.noise_const':
        lambda r: f'noise{int(np.log2(int(r)))*2-4}',
    r'synthesis\.b(\d+)\.conv1\.noise_strength':
        lambda r: f'{r}x{r}/Conv1/noise_strength',
    r'synthesis\.b(\d+)\.conv1\.affine\.weight':
        lambda r: f'{r}x{r}/Conv1/mod_weight',
    r'synthesis\.b(\d+)\.conv1\.affine\.bias':
        lambda r: f'{r}x{r}/Conv1/mod_bias',
    r'synthesis\.b(\d+)\.torgb\.weight':
        lambda r: f'{r}x{r}/ToRGB/weight',
    r'synthesis\.b(\d+)\.torgb\.bias':
        lambda r: f'{r}x{r}/ToRGB/bias',
    r'synthesis\.b(\d+)\.torgb\.affine\.weight':
        lambda r: f'{r}x{r}/ToRGB/mod_weight',
    r'synthesis\.b(\d+)\.torgb\.affine\.bias':
        lambda r: f'{r}x{r}/ToRGB/mod_bias',
    r'synthesis\.b(\d+)\.skip\.weight':
        lambda r: f'{r}x{r}/Skip/weight',
    r'.*\.resample_filter':
        None
}
D_PTH_TO_TF_VAR_MAPPING_PATTERN = {
    r'b(\d+)\.fromrgb\.weight':
        lambda r:    f'{r}x{r}/FromRGB/weight',
    r'b(\d+)\.fromrgb\.bias':
        lambda r:    f'{r}x{r}/FromRGB/bias',
    r'b(\d+)\.conv(\d+)\.weight':
        lambda r, i: f'{r}x{r}/Conv{i}{["","_down"][int(i)]}/weight',
    r'b(\d+)\.conv(\d+)\.bias':
        lambda r, i: f'{r}x{r}/Conv{i}{["","_down"][int(i)]}/bias',
    r'b(\d+)\.skip\.weight':
        lambda r:    f'{r}x{r}/Skip/weight',
    r'mapping\.embed\.weight':
        lambda:      'LabelEmbed/weight',
    r'mapping\.embed\.bias':
        lambda:      'LabelEmbed/bias',
    r'mapping\.fc(\d+)\.weight':
        lambda i:    f'Mapping{i}/weight',
    r'mapping\.fc(\d+)\.bias':
        lambda i:    f'Mapping{i}/bias',
    r'b4\.conv\.weight':
        lambda:      '4x4/Conv/weight',
    r'b4\.conv\.bias':
        lambda:      '4x4/Conv/bias',
    r'b4\.fc\.weight':
        lambda:      '4x4/Dense0/weight',
    r'b4\.fc\.bias':
        lambda:      '4x4/Dense0/bias',
    r'b4\.out\.weight':
        lambda:      'Output/weight',
    r'b4\.out\.bias':
        lambda:      'Output/bias',
    r'.*\.resample_filter':
        None
}


class StyleGAN2ADAPTHConverter(BaseConverter):
    """Defines the converter for converting StyleGAN2-ADA (PyTorch) model."""

    def load_source(self, path):
        sys.path.insert(0, CODE_PATH)
        with open(path, 'rb') as f:
            model = pickle.load(f)
        sys.path.pop(0)
        self.src_models['generator'] = model['G']
        self.src_models['discriminator'] = model['D']
        self.src_models['generator_smooth'] = model['G_ema']

    def parse_model_config(self):
        G = self.src_models['generator']
        z_dim = G.z_dim
        label_dim = G.c_dim
        w_dim = G.w_dim
        image_channels = G.img_channels
        resolution = G.img_resolution

        G_mapping_layers = G.mapping.num_layers
        G_normalize_z = True
        G_embedding_bias = True
        G_embedding_use_wscale = True
        G_normalize_embedding = True
        G_normalize_embedding_latent = False
        G_arch = G.synthesis.b4.architecture.replace('orig', 'origin')
        G_conv_clamp = G.synthesis.b4.conv1.conv_clamp

        D = self.src_models['discriminator']
        D_arch = D.b4.architecture.replace('orig', 'origin')
        D_conv_clamp = D.b4.conv.conv_clamp
        if label_dim > 0:
            D_mapping_layers = D.mapping.num_layers
            D_embedding_dim = D.mapping.w_dim
        else:
            D_mapping_layers = 0
            D_embedding_dim = 0
        D_embedding_bias = True
        D_embedding_use_wscale = True
        D_normalize_embedding = True

        self.dst_kwargs['generator'] = dict(
            model_type='StyleGAN2Generator',
            resolution=resolution,
            z_dim=z_dim,
            w_dim=w_dim,
            normalize_z=G_normalize_z,
            mapping_layers=G_mapping_layers,
            embedding_bias=G_embedding_bias,
            embedding_use_wscale=G_embedding_use_wscale,
            normalize_embedding=G_normalize_embedding,
            normalize_embedding_latent=G_normalize_embedding_latent,
            label_dim=label_dim,
            image_channels=image_channels,
            architecture=G_arch,
            conv_clamp=G_conv_clamp)
        self.dst_kwargs['discriminator'] = dict(
            model_type='StyleGAN2Discriminator',
            resolution=resolution,
            label_dim=label_dim,
            image_channels=image_channels,
            architecture=D_arch,
            embedding_dim=D_embedding_dim,
            embedding_bias=D_embedding_bias,
            embedding_use_wscale=D_embedding_use_wscale,
            normalize_embedding=D_normalize_embedding,
            mapping_layers=D_mapping_layers,
            conv_clamp=D_conv_clamp)
        self.dst_kwargs['generator_smooth'] = deepcopy(
            self.dst_kwargs['generator'])

    @staticmethod
    def convert_generator(src_model, dst_model, log_fn=None):
        """Converts the generator weights."""
        # Get source weights.
        src_vars = dict(src_model.named_parameters())
        src_vars.update(dict(src_model.named_buffers()))
        # Get target weights.
        dst_state = deepcopy(dst_model.state_dict())
        # Get variable mapping.
        official_tf_to_pth_var_mapping = {}
        for name in src_vars.keys():
            for pattern, fn in G_PTH_TO_TF_VAR_MAPPING_PATTERN.items():
                match = re.fullmatch(pattern, name)
                if match:
                    if fn is not None:
                        official_tf_to_pth_var_mapping[
                            fn(*match.groups())] = name
                    break
        dst_to_src_mapping = dst_model.pth_to_tf_var_mapping
        # Convert.
        for dst_name, tf_name in dst_to_src_mapping.items():
            assert tf_name in official_tf_to_pth_var_mapping
            src_name = official_tf_to_pth_var_mapping[tf_name]
            assert src_name in src_vars, f'Var `{src_name}` missing.'
            assert dst_name in dst_state, f'Var `{dst_name}` missing.'
            if log_fn is not None:
                log_fn(f'Converting `{src_name}` to `{dst_name}`.',
                       indent_level=2, is_verbose=True)
            var = deepcopy(src_vars[src_name].data.cpu())
            if 'Const' in tf_name:
                var = var.unsqueeze(0)
            if 'noise' in tf_name and 'noise_' not in tf_name:
                var = var.unsqueeze(0).unsqueeze(0)
            dst_state[dst_name] = var
        return dst_state

    @staticmethod
    def convert_discriminator(src_model, dst_model, log_fn=None):
        """Converts the discriminator weights."""
        # Get source weights.
        src_vars = dict(src_model.named_parameters())
        src_vars.update(dict(src_model.named_buffers()))
        # Get target weights.
        dst_state = deepcopy(dst_model.state_dict())
        # Get variable mapping.
        official_tf_to_pth_var_mapping = {}
        for name in src_vars.keys():
            for pattern, fn in D_PTH_TO_TF_VAR_MAPPING_PATTERN.items():
                match = re.fullmatch(pattern, name)
                if match:
                    if fn is not None:
                        official_tf_to_pth_var_mapping[
                            fn(*match.groups())] = name
                    break
        dst_to_src_mapping = dst_model.pth_to_tf_var_mapping
        # Convert.
        for dst_name, tf_name in dst_to_src_mapping.items():
            assert tf_name in official_tf_to_pth_var_mapping
            src_name = official_tf_to_pth_var_mapping[tf_name]
            assert src_name in src_vars, f'Var `{src_name}` missing.'
            assert dst_name in dst_state, f'Var `{dst_name}` missing.'
            if log_fn is not None:
                log_fn(f'Converting `{src_name}` to `{dst_name}`.',
                       indent_level=2, is_verbose=True)
            var = deepcopy(src_vars[src_name].data.cpu())
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
        G_src.eval().cuda()
        D_src.eval().cuda()
        Gs_src.eval().cuda()
        G_dst.eval().cuda()
        D_dst.eval().cuda()
        Gs_dst.eval().cuda()
        latent_dim = G_dst.latent_dim
        label_dim = G_dst.label_dim
        fp16_res = None
        if not FORCE_FP32:
            fp16_res = max(G_dst.resolution // (2 ** (NUM_FP16_RES - 1)), 8)

        gs_error = 0.0  # The error of Gs(z) between source and target.
        gs_mean = 0.0  # The mean of Gs(z) from source.
        dg_error = 0.0  # The error of D(G(z)) between source and target.
        dg_mean = 0.0  # The mean of D(G(z)) from source.
        for i in range(num):
            ##### Test Gs(z) #####
            # Latent code.
            latent = np.random.randn(1, *latent_dim)
            latent = torch.from_numpy(latent).type(torch.FloatTensor).cuda()
            # Label.
            if label_dim:
                label = np.zeros((1, label_dim), np.float32)
                label_id = np.random.randint(label_dim)
                label[0, label_id] = 1.0
                label = torch.from_numpy(label).to(latent)
            else:
                label = None
            with torch.no_grad():
                # Forward source.
                src_image = Gs_src(latent, label,
                                   truncation_psi=TRUNC_PSI,
                                   truncation_cutoff=TRUNC_LAYERS,
                                   noise_mode='const',
                                   fused_modconv=FUSED_MODULATE,
                                   force_fp32=FORCE_FP32)
                # Forward target.
                dst_image = Gs_dst(latent, label,
                                   trunc_psi=TRUNC_PSI,
                                   trunc_layers=TRUNC_LAYERS,
                                   noise_mode='const',
                                   fused_modulate=FUSED_MODULATE,
                                   fp16_res=fp16_res,
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
                src_image = src_image.detach().cpu().numpy()
                html.set_cell(i, 1, image=postprocess_image(src_image)[0])
                dst_image = dst_image.detach().cpu().numpy()
                html.set_cell(i, 2, image=postprocess_image(dst_image)[0])

            ##### Test D(G(z)) #####
            # Latent code.
            latent = np.random.randn(1, *latent_dim)
            latent = torch.from_numpy(latent).type(torch.FloatTensor).cuda()
            # Label.
            if label_dim:
                label = np.zeros((1, label_dim), np.float32)
                label_id = np.random.randint(label_dim)
                label[0, label_id] = 1.0
                label = torch.from_numpy(label).to(latent)
            else:
                label = None
            with torch.no_grad():
                # Forward source.
                src_image = G_src(latent, label,
                                  truncation_psi=TRUNC_PSI,
                                  truncation_cutoff=TRUNC_LAYERS,
                                  noise_mode='const',
                                  fused_modconv=FUSED_MODULATE,
                                  force_fp32=FORCE_FP32)
                src_score = D_src(src_image, label, force_fp32=FORCE_FP32)
                # Forward target.
                dst_image = G_dst(latent, label,
                                  trunc_psi=TRUNC_PSI,
                                  trunc_layers=TRUNC_LAYERS,
                                  noise_mode='const',
                                  fused_modulate=FUSED_MODULATE,
                                  fp16_res=fp16_res,
                                  impl=IMPL)['image']
                dst_score = D_dst(
                    dst_image, label, fp16_res=fp16_res, impl=IMPL)['score']
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
        fp16_res = None
        if not FORCE_FP32:
            fp16_res = max(G_dst.resolution // (2 ** (NUM_FP16_RES - 1)), 8)

        # Build optimizer for source model.
        G_src_opt = torch.optim.SGD(G_src.parameters(),
                                    lr=learning_rate,
                                    momentum=0.0,
                                    weight_decay=0.0)
        D_src_opt = torch.optim.SGD(D_src.parameters(),
                                    lr=learning_rate,
                                    momentum=0.0,
                                    weight_decay=0.0)

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
            temp_src_model = deepcopy(G_src).eval().cpu()
            temp_dst_model = deepcopy(G_dst).eval().cpu()
            temp_src_state = self.convert_generator(
                temp_src_model, temp_dst_model)
            temp_dst_state = temp_dst_model.state_dict()
            self.check_weight(temp_src_state, temp_dst_state)
        self.logger.print('Checking discriminator ...', indent_level=2)
        with torch.no_grad():
            temp_src_model = deepcopy(D_src).eval().cpu()
            temp_dst_model = deepcopy(D_dst).eval().cpu()
            temp_src_state = self.convert_discriminator(
                temp_src_model, temp_dst_model)
            temp_dst_state = temp_dst_model.state_dict()
            self.check_weight(temp_src_state, temp_dst_state)

        for i in range(num):
            ##### Train discriminator #####
            # Latent code.
            latent = np.random.randn(1, *latent_dim)
            latent = torch.from_numpy(latent).type(torch.FloatTensor).cuda()
            # Label.
            if label_dim:
                label = np.zeros((1, label_dim), np.float32)
                label_id = np.random.randint(label_dim)
                label[0, label_id] = 1.0
                label = torch.from_numpy(label).to(latent)
            else:
                label = None
            # Train source.
            for param in G_src.parameters():
                param.requires_grad = False
            for param in D_src.parameters():
                param.requires_grad = True
            src_ws = G_src.mapping(latent, label,
                                   skip_w_avg_update=True)
            src_image = G_src.synthesis(src_ws,
                                        noise_mode='const',
                                        fused_modconv=FUSED_MODULATE,
                                        force_fp32=FORCE_FP32)
            src_score = D_src(src_image, label, force_fp32=FORCE_FP32)
            D_src_loss = torch.nn.functional.softplus(src_score).mean()
            D_src_opt.zero_grad()
            D_src_loss.backward()
            D_src_opt.step()
            # Train target.
            for param in G_dst.parameters():
                param.requires_grad = False
            for param in D_dst.parameters():
                param.requires_grad = True
            dst_image = G_dst(latent, label,
                              w_moving_decay=1.0,
                              style_mixing_prob=0,
                              noise_mode='const',
                              fused_modulate=FUSED_MODULATE,
                              fp16_res=fp16_res,
                              impl=IMPL)['image']
            dst_score = D_dst(
                dst_image, label, fp16_res=fp16_res, impl=IMPL)['score']
            D_dst_loss = torch.nn.functional.softplus(dst_score).mean()
            D_dst_opt.zero_grad()
            D_dst_loss.backward()
            D_dst_opt.step()
            # Compare.
            self.logger.print(f'Step {i:03d}, train discriminator ... ('
                              f'source score {src_score[0][0].item():.6e}, '
                              f'target score {dst_score[0][0].item():.6e}, '
                              f'source loss {D_src_loss.item():.6e}, '
                              f'target loss {D_dst_loss.item():.6e})',
                              indent_level=1)
            self.logger.print('Checking generator ...', indent_level=2)
            with torch.no_grad():
                temp_src_model = deepcopy(G_src).eval().cpu()
                temp_dst_model = deepcopy(G_dst).eval().cpu()
                temp_src_state = self.convert_generator(
                    temp_src_model, temp_dst_model)
                temp_dst_state = temp_dst_model.state_dict()
                init_state = self.dst_states['generator']
                self.check_weight(temp_src_state, temp_dst_state, init_state)
            self.logger.print('Checking discriminator ...', indent_level=2)
            with torch.no_grad():
                temp_src_model = deepcopy(D_src).eval().cpu()
                temp_dst_model = deepcopy(D_dst).eval().cpu()
                temp_src_state = self.convert_discriminator(
                    temp_src_model, temp_dst_model)
                temp_dst_state = temp_dst_model.state_dict()
                init_state = self.dst_states['discriminator']
                self.check_weight(temp_src_state, temp_dst_state, init_state)

            ##### Train generator #####
            # Latent code.
            latent = np.random.randn(1, *latent_dim)
            latent = torch.from_numpy(latent).type(torch.FloatTensor).cuda()
            # Label.
            if label_dim:
                label = np.zeros((1, label_dim), np.float32)
                label_id = np.random.randint(label_dim)
                label[0, label_id] = 1.0
                label = torch.from_numpy(label).to(latent)
            else:
                label = None
            # Train source.
            for param in G_src.parameters():
                param.requires_grad = True
            for param in D_src.parameters():
                param.requires_grad = False
            src_ws = G_src.mapping(latent, label,
                                   skip_w_avg_update=True)
            src_image = G_src.synthesis(src_ws,
                                        noise_mode='const',
                                        fused_modconv=False,
                                        force_fp32=FORCE_FP32)
            src_score = D_src(src_image, label, force_fp32=FORCE_FP32)
            G_src_loss = -torch.nn.functional.softplus(src_score).mean()
            G_src_opt.zero_grad()
            G_src_loss.backward()
            G_src_opt.step()
            # Train target.
            for param in G_dst.parameters():
                param.requires_grad = True
            for param in D_dst.parameters():
                param.requires_grad = False
            dst_image = G_dst(latent, label,
                              w_moving_decay=1.0,
                              style_mixing_prob=0,
                              noise_mode='const',
                              fused_modulate=False,
                              fp16_res=fp16_res,
                              impl=IMPL)['image']
            dst_score = D_dst(
                dst_image, label, fp16_res=fp16_res, impl=IMPL)['score']
            G_dst_loss = -torch.nn.functional.softplus(dst_score).mean()
            G_dst_opt.zero_grad()
            G_dst_loss.backward()
            G_dst_opt.step()
            # Compare.
            self.logger.print(f'Step {i:03d}, train generator ... ('
                              f'source score {src_score[0][0].item():.6e}, '
                              f'target score {dst_score[0][0].item():.6e}, '
                              f'source loss {G_src_loss.item():.6e}, '
                              f'target loss {G_dst_loss.item():.6e})',
                              indent_level=1)
            self.logger.print('Checking generator ...', indent_level=2)
            with torch.no_grad():
                temp_src_model = deepcopy(G_src).eval().cpu()
                temp_dst_model = deepcopy(G_dst).eval().cpu()
                temp_src_state = self.convert_generator(
                    temp_src_model, temp_dst_model)
                temp_dst_state = temp_dst_model.state_dict()
                init_state = self.dst_states['generator']
                self.check_weight(temp_src_state, temp_dst_state, init_state)
            self.logger.print('Checking discriminator ...', indent_level=2)
            with torch.no_grad():
                temp_src_model = deepcopy(D_src).eval().cpu()
                temp_dst_model = deepcopy(D_dst).eval().cpu()
                temp_src_state = self.convert_discriminator(
                    temp_src_model, temp_dst_model)
                temp_dst_state = temp_dst_model.state_dict()
                init_state = self.dst_states['discriminator']
                self.check_weight(temp_src_state, temp_dst_state, init_state)
