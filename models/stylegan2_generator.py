# python3.7
"""Contains the implementation of generator described in StyleGAN2.

Compared to that of StyleGAN, the generator in StyleGAN2 mainly introduces style
demodulation, adds skip connections, increases model size, and disables
progressive growth. This script ONLY supports config F in the original paper.

Paper: https://arxiv.org/pdf/1912.04958.pdf

Official TensorFlow implementation: https://github.com/NVlabs/stylegan2
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from third_party.stylegan2_official_ops import fma
from third_party.stylegan2_official_ops import bias_act
from third_party.stylegan2_official_ops import upfirdn2d
from third_party.stylegan2_official_ops import conv2d_gradfix
from .utils.ops import all_gather

__all__ = ['StyleGAN2Generator']

# Resolutions allowed.
_RESOLUTIONS_ALLOWED = [8, 16, 32, 64, 128, 256, 512, 1024]

# Architectures allowed.
_ARCHITECTURES_ALLOWED = ['resnet', 'skip', 'origin']

# pylint: disable=missing-function-docstring

class StyleGAN2Generator(nn.Module):
    """Defines the generator network in StyleGAN2.

    NOTE: The synthesized images are with `RGB` channel order and pixel range
    [-1, 1].

    Settings for the mapping network:

    (1) z_dim: Dimension of the input latent space, Z. (default: 512)
    (2) w_dim: Dimension of the output latent space, W. (default: 512)
    (3) repeat_w: Repeat w-code for different layers. (default: True)
    (4) normalize_z: Whether to normalize the z-code. (default: True)
    (5) mapping_layers: Number of layers of the mapping network. (default: 8)
    (6) mapping_fmaps: Number of hidden channels of the mapping network.
        (default: 512)
    (7) mapping_use_wscale: Whether to use weight scaling for the mapping
        network. (default: True)
    (8) mapping_wscale_gain: The factor to control weight scaling for the
        mapping network (default: 1.0)
    (9) mapping_lr_mul: Learning rate multiplier for the mapping network.
        (default: 0.01)

    Settings for conditional generation:

    (1) label_dim: Dimension of the additional label for conditional generation.
        In one-hot conditioning case, it is equal to the number of classes. If
        set to 0, conditioning training will be disabled. (default: 0)
    (2) embedding_dim: Dimension of the embedding space, if needed.
        (default: 512)
    (3) embedding_bias: Whether to add bias to embedding learning.
        (default: True)
    (4) embedding_use_wscale: Whether to use weight scaling for embedding
        learning. (default: True)
    (5) embedding_wscale_gain: The factor to control weight scaling for
        embedding. (default: 1.0)
    (6) embedding_lr_mul: Learning rate multiplier for the embedding learning.
        (default: 1.0)
    (7) normalize_embedding: Whether to normalize the embedding. (default: True)
    (8) normalize_embedding_latent: Whether to normalize the embedding together
        with the latent. (default: False)

    Settings for the synthesis network:

    (1) resolution: The resolution of the output image. (default: -1)
    (2) init_res: The initial resolution to start with convolution. (default: 4)
    (3) image_channels: Number of channels of the output image. (default: 3)
    (4) final_tanh: Whether to use `tanh` to control the final pixel range.
        (default: False)
    (5) const_input: Whether to use a constant in the first convolutional layer.
        (default: True)
    (6) architecture: Type of architecture. Support `origin`, `skip`, and
        `resnet`. (default: `skip`)
    (7) demodulate: Whether to perform style demodulation. (default: True)
    (8) use_wscale: Whether to use weight scaling. (default: True)
    (9) wscale_gain: The factor to control weight scaling. (default: 1.0)
    (10) lr_mul: Learning rate multiplier for the synthesis network.
         (default: 1.0)
    (11) noise_type: Type of noise added to the convolutional results at each
         layer. (default: `spatial`)
    (12) fmaps_base: Factor to control number of feature maps for each layer.
         (default: 32 << 10)
    (13) fmaps_max: Maximum number of feature maps in each layer. (default: 512)
    (14) filter_kernel: Kernel used for filtering (e.g., downsampling).
         (default: (1, 3, 3, 1))
    (15) conv_clamp: A threshold to clamp the output of convolution layers to
         avoid overflow under FP16 training. (default: None)
    (16) eps: A small value to avoid divide overflow. (default: 1e-8)

    Runtime settings:

    (1) w_moving_decay: Decay factor for updating `w_avg`, which is used for
        training only. Set `None` to disable. (default: None)
    (2) sync_w_avg: Synchronizing the stats of `w_avg` across replicas. If set
        as `True`, the stats will be more accurate, yet the speed maybe a little
        bit slower. (default: False)
    (3) style_mixing_prob: Probability to perform style mixing as a training
        regularization. Set `None` to disable. (default: None)
    (4) trunc_psi: Truncation psi, set `None` to disable. (default: None)
    (5) trunc_layers: Number of layers to perform truncation. (default: None)
    (6) noise_mode: Mode of the layer-wise noise. Support `none`, `random`,
        `const`. (default: `const`)
    (7) fused_modulate: Whether to fuse `style_modulate` and `conv2d` together.
        (default: False)
    (8) fp16_res: Layers at resolution higher than (or equal to) this field will
        use `float16` precision for computation. This is merely used for
        acceleration. If set as `None`, all layers will use `float32` by
        default. (default: None)
    (9) impl: Implementation mode of some particular ops, e.g., `filtering`,
        `bias_act`, etc. `cuda` means using the official CUDA implementation
        from StyleGAN2, while `ref` means using the native PyTorch ops.
        (default: `cuda`)
    """

    def __init__(self,
                 # Settings for mapping network.
                 z_dim=512,
                 w_dim=512,
                 repeat_w=True,
                 normalize_z=True,
                 mapping_layers=8,
                 mapping_fmaps=512,
                 mapping_use_wscale=True,
                 mapping_wscale_gain=1.0,
                 mapping_lr_mul=0.01,
                 # Settings for conditional generation.
                 label_dim=0,
                 embedding_dim=512,
                 embedding_bias=True,
                 embedding_use_wscale=True,
                 embedding_wscale_gian=1.0,
                 embedding_lr_mul=1.0,
                 normalize_embedding=True,
                 normalize_embedding_latent=False,
                 # Settings for synthesis network.
                 resolution=-1,
                 init_res=4,
                 image_channels=3,
                 final_tanh=False,
                 const_input=True,
                 architecture='skip',
                 demodulate=True,
                 use_wscale=True,
                 wscale_gain=1.0,
                 lr_mul=1.0,
                 noise_type='spatial',
                 fmaps_base=32 << 10,
                 fmaps_max=512,
                 filter_kernel=(1, 3, 3, 1),
                 conv_clamp=None,
                 eps=1e-8):
        """Initializes with basic settings.

        Raises:
            ValueError: If the `resolution` is not supported, or `architecture`
                is not supported.
        """
        super().__init__()

        if resolution not in _RESOLUTIONS_ALLOWED:
            raise ValueError(f'Invalid resolution: `{resolution}`!\n'
                             f'Resolutions allowed: {_RESOLUTIONS_ALLOWED}.')
        architecture = architecture.lower()
        if architecture not in _ARCHITECTURES_ALLOWED:
            raise ValueError(f'Invalid architecture: `{architecture}`!\n'
                             f'Architectures allowed: '
                             f'{_ARCHITECTURES_ALLOWED}.')

        self.z_dim = z_dim
        self.w_dim = w_dim
        self.repeat_w = repeat_w
        self.normalize_z = normalize_z
        self.mapping_layers = mapping_layers
        self.mapping_fmaps = mapping_fmaps
        self.mapping_use_wscale = mapping_use_wscale
        self.mapping_wscale_gain = mapping_wscale_gain
        self.mapping_lr_mul = mapping_lr_mul

        self.label_dim = label_dim
        self.embedding_dim = embedding_dim
        self.embedding_bias = embedding_bias
        self.embedding_use_wscale = embedding_use_wscale
        self.embedding_wscale_gain = embedding_wscale_gian
        self.embedding_lr_mul = embedding_lr_mul
        self.normalize_embedding = normalize_embedding
        self.normalize_embedding_latent = normalize_embedding_latent

        self.resolution = resolution
        self.init_res = init_res
        self.image_channels = image_channels
        self.final_tanh = final_tanh
        self.const_input = const_input
        self.architecture = architecture
        self.demodulate = demodulate
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.lr_mul = lr_mul
        self.noise_type = noise_type.lower()
        self.fmaps_base = fmaps_base
        self.fmaps_max = fmaps_max
        self.filter_kernel = filter_kernel
        self.conv_clamp = conv_clamp
        self.eps = eps

        # Dimension of latent space, which is convenient for sampling.
        self.latent_dim = (z_dim,)

        # Number of synthesis (convolutional) layers.
        self.num_layers = int(np.log2(resolution // init_res * 2)) * 2

        self.mapping = MappingNetwork(
            input_dim=z_dim,
            output_dim=w_dim,
            num_outputs=self.num_layers,
            repeat_output=repeat_w,
            normalize_input=normalize_z,
            num_layers=mapping_layers,
            hidden_dim=mapping_fmaps,
            use_wscale=mapping_use_wscale,
            wscale_gain=mapping_wscale_gain,
            lr_mul=mapping_lr_mul,
            label_dim=label_dim,
            embedding_dim=embedding_dim,
            embedding_bias=embedding_bias,
            embedding_use_wscale=embedding_use_wscale,
            embedding_wscale_gian=embedding_wscale_gian,
            embedding_lr_mul=embedding_lr_mul,
            normalize_embedding=normalize_embedding,
            normalize_embedding_latent=normalize_embedding_latent,
            eps=eps)

        # This is used for truncation trick.
        if self.repeat_w:
            self.register_buffer('w_avg', torch.zeros(w_dim))
        else:
            self.register_buffer('w_avg', torch.zeros(self.num_layers * w_dim))

        self.synthesis = SynthesisNetwork(resolution=resolution,
                                          init_res=init_res,
                                          w_dim=w_dim,
                                          image_channels=image_channels,
                                          final_tanh=final_tanh,
                                          const_input=const_input,
                                          architecture=architecture,
                                          demodulate=demodulate,
                                          use_wscale=use_wscale,
                                          wscale_gain=wscale_gain,
                                          lr_mul=lr_mul,
                                          noise_type=noise_type,
                                          fmaps_base=fmaps_base,
                                          filter_kernel=filter_kernel,
                                          fmaps_max=fmaps_max,
                                          conv_clamp=conv_clamp,
                                          eps=eps)

        self.pth_to_tf_var_mapping = {'w_avg': 'dlatent_avg'}
        for key, val in self.mapping.pth_to_tf_var_mapping.items():
            self.pth_to_tf_var_mapping[f'mapping.{key}'] = val
        for key, val in self.synthesis.pth_to_tf_var_mapping.items():
            self.pth_to_tf_var_mapping[f'synthesis.{key}'] = val

    def set_space_of_latent(self, space_of_latent):
        """Sets the space to which the latent code belong.

        See `SynthesisNetwork` for more details.
        """
        self.synthesis.set_space_of_latent(space_of_latent)

    def forward(self,
                z,
                label=None,
                w_moving_decay=None,
                sync_w_avg=False,
                style_mixing_prob=None,
                trunc_psi=None,
                trunc_layers=None,
                noise_mode='const',
                fused_modulate=False,
                fp16_res=None,
                impl='cuda'):
        """Connects mapping network and synthesis network.

        This forward function will also update the average `w_code`, perform
        style mixing as a training regularizer, and do truncation trick, which
        is specially designed for inference.

        Concretely, the truncation trick acts as follows:

        For layers in range [0, truncation_layers), the truncated w-code is
        computed as

        w_new = w_avg + (w - w_avg) * trunc_psi

        To disable truncation, please set

        (1) trunc_psi = 1.0 (None) OR
        (2) trunc_layers = 0 (None)
        """

        mapping_results = self.mapping(z, label, impl=impl)

        w = mapping_results['w']
        if self.training and w_moving_decay is not None:
            if sync_w_avg:
                batch_w_avg = all_gather(w.detach()).mean(dim=0)
            else:
                batch_w_avg = w.detach().mean(dim=0)
            self.w_avg.copy_(batch_w_avg.lerp(self.w_avg, w_moving_decay))

        wp = mapping_results.pop('wp')
        if self.training and style_mixing_prob is not None:
            if np.random.uniform() < style_mixing_prob:
                new_z = torch.randn_like(z)
                new_wp = self.mapping(new_z, label, impl=impl)['wp']
                mixing_cutoff = np.random.randint(1, self.num_layers)
                wp[:, mixing_cutoff:] = new_wp[:, mixing_cutoff:]

        if not self.training:
            trunc_psi = 1.0 if trunc_psi is None else trunc_psi
            trunc_layers = 0 if trunc_layers is None else trunc_layers
            if trunc_psi < 1.0 and trunc_layers > 0:
                w_avg = self.w_avg.reshape(1, -1, self.w_dim)[:, :trunc_layers]
                wp[:, :trunc_layers] = w_avg.lerp(
                    wp[:, :trunc_layers], trunc_psi)

        synthesis_results = self.synthesis(wp,
                                           noise_mode=noise_mode,
                                           fused_modulate=fused_modulate,
                                           impl=impl,
                                           fp16_res=None)

        return {**mapping_results, **synthesis_results}


class MappingNetwork(nn.Module):
    """Implements the latent space mapping network.

    Basically, this network executes several dense layers in sequence, and the
    label embedding if needed.
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 num_outputs,
                 repeat_output,
                 normalize_input,
                 num_layers,
                 hidden_dim,
                 use_wscale,
                 wscale_gain,
                 lr_mul,
                 label_dim,
                 embedding_dim,
                 embedding_bias,
                 embedding_use_wscale,
                 embedding_wscale_gian,
                 embedding_lr_mul,
                 normalize_embedding,
                 normalize_embedding_latent,
                 eps):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_outputs = num_outputs
        self.repeat_output = repeat_output
        self.normalize_input = normalize_input
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.lr_mul = lr_mul
        self.label_dim = label_dim
        self.embedding_dim = embedding_dim
        self.embedding_bias = embedding_bias
        self.embedding_use_wscale = embedding_use_wscale
        self.embedding_wscale_gian = embedding_wscale_gian
        self.embedding_lr_mul = embedding_lr_mul
        self.normalize_embedding = normalize_embedding
        self.normalize_embedding_latent = normalize_embedding_latent
        self.eps = eps

        self.pth_to_tf_var_mapping = {}

        self.norm = PixelNormLayer(dim=1, eps=eps)

        if self.label_dim > 0:
            input_dim = input_dim + embedding_dim
            self.embedding = DenseLayer(in_channels=label_dim,
                                        out_channels=embedding_dim,
                                        add_bias=embedding_bias,
                                        init_bias=0.0,
                                        use_wscale=embedding_use_wscale,
                                        wscale_gain=embedding_wscale_gian,
                                        lr_mul=embedding_lr_mul,
                                        activation_type='linear')
            self.pth_to_tf_var_mapping['embedding.weight'] = 'LabelEmbed/weight'
            if self.embedding_bias:
                self.pth_to_tf_var_mapping['embedding.bias'] = 'LabelEmbed/bias'

        if num_outputs is not None and not repeat_output:
            output_dim = output_dim * num_outputs
        for i in range(num_layers):
            in_channels = (input_dim if i == 0 else hidden_dim)
            out_channels = (output_dim if i == (num_layers - 1) else hidden_dim)
            self.add_module(f'dense{i}',
                            DenseLayer(in_channels=in_channels,
                                       out_channels=out_channels,
                                       add_bias=True,
                                       init_bias=0.0,
                                       use_wscale=use_wscale,
                                       wscale_gain=wscale_gain,
                                       lr_mul=lr_mul,
                                       activation_type='lrelu'))
            self.pth_to_tf_var_mapping[f'dense{i}.weight'] = f'Dense{i}/weight'
            self.pth_to_tf_var_mapping[f'dense{i}.bias'] = f'Dense{i}/bias'

    def forward(self, z, label=None, impl='cuda'):
        if z.ndim != 2 or z.shape[1] != self.input_dim:
            raise ValueError(f'Input latent code should be with shape '
                             f'[batch_size, input_dim], where '
                             f'`input_dim` equals to {self.input_dim}!\n'
                             f'But `{z.shape}` is received!')
        if self.normalize_input:
            z = self.norm(z)

        if self.label_dim > 0:
            if label is None:
                raise ValueError(f'Model requires an additional label '
                                 f'(with dimension {self.label_dim}) as input, '
                                 f'but no label is received!')
            if label.ndim != 2 or label.shape != (z.shape[0], self.label_dim):
                raise ValueError(f'Input label should be with shape '
                                 f'[batch_size, label_dim], where '
                                 f'`batch_size` equals to that of '
                                 f'latent codes ({z.shape[0]}) and '
                                 f'`label_dim` equals to {self.label_dim}!\n'
                                 f'But `{label.shape}` is received!')
            label = label.to(dtype=torch.float32)
            embedding = self.embedding(label, impl=impl)
            if self.normalize_embedding:
                embedding = self.norm(embedding)
            w = torch.cat((z, embedding), dim=1)
        else:
            w = z

        if self.label_dim > 0 and self.normalize_embedding_latent:
            w = self.norm(w)

        for i in range(self.num_layers):
            w = getattr(self, f'dense{i}')(w, impl=impl)

        wp = None
        if self.num_outputs is not None:
            if self.repeat_output:
                wp = w.unsqueeze(1).repeat((1, self.num_outputs, 1))
            else:
                wp = w.reshape(-1, self.num_outputs, self.output_dim)

        results = {
            'z': z,
            'label': label,
            'w': w,
            'wp': wp,
        }
        if self.label_dim > 0:
            results['embedding'] = embedding
        return results


class SynthesisNetwork(nn.Module):
    """Implements the image synthesis network.

    Basically, this network executes several convolutional layers in sequence.
    """

    def __init__(self,
                 resolution,
                 init_res,
                 w_dim,
                 image_channels,
                 final_tanh,
                 const_input,
                 architecture,
                 demodulate,
                 use_wscale,
                 wscale_gain,
                 lr_mul,
                 noise_type,
                 fmaps_base,
                 fmaps_max,
                 filter_kernel,
                 conv_clamp,
                 eps):
        super().__init__()

        self.init_res = init_res
        self.init_res_log2 = int(np.log2(init_res))
        self.resolution = resolution
        self.final_res_log2 = int(np.log2(resolution))
        self.w_dim = w_dim
        self.image_channels = image_channels
        self.final_tanh = final_tanh
        self.const_input = const_input
        self.architecture = architecture.lower()
        self.demodulate = demodulate
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.lr_mul = lr_mul
        self.noise_type = noise_type.lower()
        self.fmaps_base = fmaps_base
        self.fmaps_max = fmaps_max
        self.filter_kernel = filter_kernel
        self.conv_clamp = conv_clamp
        self.eps = eps

        self.num_layers = (self.final_res_log2 - self.init_res_log2 + 1) * 2

        self.pth_to_tf_var_mapping = {}

        for res_log2 in range(self.init_res_log2, self.final_res_log2 + 1):
            res = 2 ** res_log2
            in_channels = self.get_nf(res // 2)
            out_channels = self.get_nf(res)
            block_idx = res_log2 - self.init_res_log2

            # Early layer.
            if res == init_res:
                if self.const_input:
                    self.add_module('early_layer',
                                    InputLayer(init_res=res,
                                               channels=out_channels))
                    self.pth_to_tf_var_mapping['early_layer.const'] = (
                        f'{res}x{res}/Const/const')
                else:
                    channels = out_channels * res * res
                    self.add_module('early_layer',
                                    DenseLayer(in_channels=w_dim,
                                               out_channels=channels,
                                               add_bias=True,
                                               init_bias=0.0,
                                               use_wscale=use_wscale,
                                               wscale_gain=wscale_gain,
                                               lr_mul=lr_mul,
                                               activation_type='lrelu'))
                    self.pth_to_tf_var_mapping['early_layer.weight'] = (
                        f'{res}x{res}/Dense/weight')
                    self.pth_to_tf_var_mapping['early_layer.bias'] = (
                        f'{res}x{res}/Dense/bias')
            else:
                # Residual branch (kernel 1x1) with upsampling, without bias,
                # with linear activation.
                if self.architecture == 'resnet':
                    layer_name = f'residual{block_idx}'
                    self.add_module(layer_name,
                                    ConvLayer(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=1,
                                              add_bias=False,
                                              scale_factor=2,
                                              filter_kernel=filter_kernel,
                                              use_wscale=use_wscale,
                                              wscale_gain=wscale_gain,
                                              lr_mul=lr_mul,
                                              activation_type='linear',
                                              conv_clamp=None))
                    self.pth_to_tf_var_mapping[f'{layer_name}.weight'] = (
                        f'{res}x{res}/Skip/weight')

                # First layer (kernel 3x3) with upsampling.
                layer_name = f'layer{2 * block_idx - 1}'
                self.add_module(layer_name,
                                ModulateConvLayer(in_channels=in_channels,
                                                  out_channels=out_channels,
                                                  resolution=res,
                                                  w_dim=w_dim,
                                                  kernel_size=3,
                                                  add_bias=True,
                                                  scale_factor=2,
                                                  filter_kernel=filter_kernel,
                                                  demodulate=demodulate,
                                                  use_wscale=use_wscale,
                                                  wscale_gain=wscale_gain,
                                                  lr_mul=lr_mul,
                                                  noise_type=noise_type,
                                                  activation_type='lrelu',
                                                  conv_clamp=conv_clamp,
                                                  eps=eps))
                self.pth_to_tf_var_mapping[f'{layer_name}.weight'] = (
                    f'{res}x{res}/Conv0_up/weight')
                self.pth_to_tf_var_mapping[f'{layer_name}.bias'] = (
                    f'{res}x{res}/Conv0_up/bias')
                self.pth_to_tf_var_mapping[f'{layer_name}.style.weight'] = (
                    f'{res}x{res}/Conv0_up/mod_weight')
                self.pth_to_tf_var_mapping[f'{layer_name}.style.bias'] = (
                    f'{res}x{res}/Conv0_up/mod_bias')
                self.pth_to_tf_var_mapping[f'{layer_name}.noise_strength'] = (
                    f'{res}x{res}/Conv0_up/noise_strength')
                self.pth_to_tf_var_mapping[f'{layer_name}.noise'] = (
                    f'noise{2 * block_idx - 1}')

            # Second layer (kernel 3x3) without upsampling.
            layer_name = f'layer{2 * block_idx}'
            self.add_module(layer_name,
                            ModulateConvLayer(in_channels=out_channels,
                                              out_channels=out_channels,
                                              resolution=res,
                                              w_dim=w_dim,
                                              kernel_size=3,
                                              add_bias=True,
                                              scale_factor=1,
                                              filter_kernel=None,
                                              demodulate=demodulate,
                                              use_wscale=use_wscale,
                                              wscale_gain=wscale_gain,
                                              lr_mul=lr_mul,
                                              noise_type=noise_type,
                                              activation_type='lrelu',
                                              conv_clamp=conv_clamp,
                                              eps=eps))
            tf_layer_name = 'Conv' if res == self.init_res else 'Conv1'
            self.pth_to_tf_var_mapping[f'{layer_name}.weight'] = (
                f'{res}x{res}/{tf_layer_name}/weight')
            self.pth_to_tf_var_mapping[f'{layer_name}.bias'] = (
                f'{res}x{res}/{tf_layer_name}/bias')
            self.pth_to_tf_var_mapping[f'{layer_name}.style.weight'] = (
                f'{res}x{res}/{tf_layer_name}/mod_weight')
            self.pth_to_tf_var_mapping[f'{layer_name}.style.bias'] = (
                f'{res}x{res}/{tf_layer_name}/mod_bias')
            self.pth_to_tf_var_mapping[f'{layer_name}.noise_strength'] = (
                f'{res}x{res}/{tf_layer_name}/noise_strength')
            self.pth_to_tf_var_mapping[f'{layer_name}.noise'] = (
                f'noise{2 * block_idx}')

            # Output convolution layer for each resolution (if needed).
            if res_log2 == self.final_res_log2 or self.architecture == 'skip':
                layer_name = f'output{block_idx}'
                self.add_module(layer_name,
                                ModulateConvLayer(in_channels=out_channels,
                                                  out_channels=image_channels,
                                                  resolution=res,
                                                  w_dim=w_dim,
                                                  kernel_size=1,
                                                  add_bias=True,
                                                  scale_factor=1,
                                                  filter_kernel=None,
                                                  demodulate=False,
                                                  use_wscale=use_wscale,
                                                  wscale_gain=wscale_gain,
                                                  lr_mul=lr_mul,
                                                  noise_type='none',
                                                  activation_type='linear',
                                                  conv_clamp=conv_clamp,
                                                  eps=eps))
                self.pth_to_tf_var_mapping[f'{layer_name}.weight'] = (
                    f'{res}x{res}/ToRGB/weight')
                self.pth_to_tf_var_mapping[f'{layer_name}.bias'] = (
                    f'{res}x{res}/ToRGB/bias')
                self.pth_to_tf_var_mapping[f'{layer_name}.style.weight'] = (
                    f'{res}x{res}/ToRGB/mod_weight')
                self.pth_to_tf_var_mapping[f'{layer_name}.style.bias'] = (
                    f'{res}x{res}/ToRGB/mod_bias')

        # Used for upsampling output images for each resolution block for sum.
        if self.architecture == 'skip':
            self.register_buffer(
                'filter', upfirdn2d.setup_filter(filter_kernel))

    def get_nf(self, res):
        """Gets number of feature maps according to the given resolution."""
        return min(self.fmaps_base // res, self.fmaps_max)

    def set_space_of_latent(self, space_of_latent):
        """Sets the space to which the latent code belong.

        This function is particularly used for choosing how to inject the latent
        code into the convolutional layers. The original generator will take a
        W-Space code and apply it for style modulation after an affine
        transformation. But, sometimes, it may need to directly feed an already
        affine-transformed code into the convolutional layer, e.g., when
        training an encoder for GAN inversion. We term the transformed space as
        Style Space (or Y-Space). This function is designed to tell the
        convolutional layers how to use the input code.

        Args:
            space_of_latent: The space to which the latent code belong. Case
                insensitive. Support `W` and `Y`.
        """
        space_of_latent = space_of_latent.upper()
        for module in self.modules():
            if isinstance(module, ModulateConvLayer):
                setattr(module, 'space_of_latent', space_of_latent)

    def forward(self,
                wp,
                noise_mode='const',
                fused_modulate=False,
                fp16_res=None,
                impl='cuda'):
        results = {'wp': wp}

        if self.const_input:
            x = self.early_layer(wp[:, 0])
        else:
            x = self.early_layer(wp[:, 0], impl=impl)

        # Cast to `torch.float16` if needed.
        if fp16_res is not None and self.init_res >= fp16_res:
            x = x.to(torch.float16)

        if self.architecture == 'origin':
            for layer_idx in range(self.num_layers - 1):
                layer = getattr(self, f'layer{layer_idx}')
                x, style = layer(x,
                                 wp[:, layer_idx],
                                 noise_mode=noise_mode,
                                 fused_modulate=fused_modulate,
                                 impl=impl)
                results[f'style{layer_idx}'] = style

                # Cast to `torch.float16` if needed.
                if layer_idx % 2 == 0 and layer_idx != self.num_layers - 2:
                    res = self.init_res * (2 ** (layer_idx // 2))
                    if fp16_res is not None and res * 2 >= fp16_res:
                        x = x.to(torch.float16)
                    else:
                        x = x.to(torch.float32)
            output_layer = getattr(self, f'output{layer_idx // 2}')
            image, style = output_layer(x,
                                        wp[:, layer_idx + 1],
                                        fused_modulate=fused_modulate,
                                        impl=impl)
            image = image.to(torch.float32)
            results[f'output_style{layer_idx // 2}'] = style

        elif self.architecture == 'skip':
            for layer_idx in range(self.num_layers - 1):
                layer = getattr(self, f'layer{layer_idx}')
                x, style = layer(x,
                                 wp[:, layer_idx],
                                 noise_mode=noise_mode,
                                 fused_modulate=fused_modulate,
                                 impl=impl)
                results[f'style{layer_idx}'] = style
                if layer_idx % 2 == 0:
                    output_layer = getattr(self, f'output{layer_idx // 2}')
                    y, style = output_layer(x,
                                            wp[:, layer_idx + 1],
                                            fused_modulate=fused_modulate,
                                            impl=impl)
                    results[f'output_style{layer_idx // 2}'] = style
                    if layer_idx == 0:
                        image = y.to(torch.float32)
                    else:
                        image = y.to(torch.float32) + upfirdn2d.upsample2d(
                            image, self.filter, impl=impl)

                    # Cast to `torch.float16` if needed.
                    if layer_idx != self.num_layers - 2:
                        res = self.init_res * (2 ** (layer_idx // 2))
                        if fp16_res is not None and res * 2 >= fp16_res:
                            x = x.to(torch.float16)
                        else:
                            x = x.to(torch.float32)

        elif self.architecture == 'resnet':
            x, style = self.layer0(x,
                                   wp[:, 0],
                                   noise_mode=noise_mode,
                                   fused_modulate=fused_modulate,
                                   impl=impl)
            results['style0'] = style
            for layer_idx in range(1, self.num_layers - 1, 2):
                # Cast to `torch.float16` if needed.
                if layer_idx % 2 == 1:
                    res = self.init_res * (2 ** (layer_idx // 2))
                    if fp16_res is not None and res * 2 >= fp16_res:
                        x = x.to(torch.float16)
                    else:
                        x = x.to(torch.float32)

                skip_layer = getattr(self, f'residual{layer_idx // 2 + 1}')
                residual = skip_layer(x, runtime_gain=np.sqrt(0.5), impl=impl)
                layer = getattr(self, f'layer{layer_idx}')
                x, style = layer(x,
                                 wp[:, layer_idx],
                                 noise_mode=noise_mode,
                                 fused_modulate=fused_modulate,
                                 impl=impl)
                results[f'style{layer_idx}'] = style
                layer = getattr(self, f'layer{layer_idx + 1}')
                x, style = layer(x,
                                 wp[:, layer_idx + 1],
                                 runtime_gain=np.sqrt(0.5),
                                 noise_mode=noise_mode,
                                 fused_modulate=fused_modulate,
                                 impl=impl)
                results[f'style{layer_idx + 1}'] = style
                x = x + residual
            output_layer = getattr(self, f'output{layer_idx // 2 + 1}')
            image, style = output_layer(x,
                                        wp[:, layer_idx + 2],
                                        fused_modulate=fused_modulate,
                                        impl=impl)
            image = image.to(torch.float32)
            results[f'output_style{layer_idx // 2}'] = style

        if self.final_tanh:
            image = torch.tanh(image)
        results['image'] = image
        return results


class PixelNormLayer(nn.Module):
    """Implements pixel-wise feature vector normalization layer."""

    def __init__(self, dim, eps):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def extra_repr(self):
        return f'dim={self.dim}, epsilon={self.eps}'

    def forward(self, x):
        scale = (x.square().mean(dim=self.dim, keepdim=True) + self.eps).rsqrt()
        return x * scale


class InputLayer(nn.Module):
    """Implements the input layer to start convolution with.

    Basically, this block starts from a const input, which is with shape
    `(channels, init_res, init_res)`.
    """

    def __init__(self, init_res, channels):
        super().__init__()
        self.const = nn.Parameter(torch.randn(1, channels, init_res, init_res))

    def forward(self, w):
        x = self.const.repeat(w.shape[0], 1, 1, 1)
        return x


class ConvLayer(nn.Module):
    """Implements the convolutional layer.

    If upsampling is needed (i.e., `scale_factor = 2`), the feature map will
    be filtered with `filter_kernel` after convolution. This layer will only be
    used for skip connection in `resnet` architecture.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 add_bias,
                 scale_factor,
                 filter_kernel,
                 use_wscale,
                 wscale_gain,
                 lr_mul,
                 activation_type,
                 conv_clamp):
        """Initializes with layer settings.

        Args:
            in_channels: Number of channels of the input tensor.
            out_channels: Number of channels of the output tensor.
            kernel_size: Size of the convolutional kernels.
            add_bias: Whether to add bias onto the convolutional result.
            scale_factor: Scale factor for upsampling.
            filter_kernel: Kernel used for filtering.
            use_wscale: Whether to use weight scaling.
            wscale_gain: Gain factor for weight scaling.
            lr_mul: Learning multiplier for both weight and bias.
            activation_type: Type of activation.
            conv_clamp: A threshold to clamp the output of convolution layers to
                avoid overflow under FP16 training.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.add_bias = add_bias
        self.scale_factor = scale_factor
        self.filter_kernel = filter_kernel
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.lr_mul = lr_mul
        self.activation_type = activation_type
        self.conv_clamp = conv_clamp

        weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        fan_in = kernel_size * kernel_size * in_channels
        wscale = wscale_gain / np.sqrt(fan_in)
        if use_wscale:
            self.weight = nn.Parameter(torch.randn(*weight_shape) / lr_mul)
            self.wscale = wscale * lr_mul
        else:
            self.weight = nn.Parameter(
                torch.randn(*weight_shape) * wscale / lr_mul)
            self.wscale = lr_mul

        if add_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
            self.bscale = lr_mul
        else:
            self.bias = None
        self.act_gain = bias_act.activation_funcs[activation_type].def_gain

        if scale_factor > 1:
            assert filter_kernel is not None
            self.register_buffer(
                'filter', upfirdn2d.setup_filter(filter_kernel))
            fh, fw = self.filter.shape
            self.filter_padding = (
                kernel_size // 2 + (fw + scale_factor - 1) // 2,
                kernel_size // 2 + (fw - scale_factor) // 2,
                kernel_size // 2 + (fh + scale_factor - 1) // 2,
                kernel_size // 2 + (fh - scale_factor) // 2)

    def extra_repr(self):
        return (f'in_ch={self.in_channels}, '
                f'out_ch={self.out_channels}, '
                f'ksize={self.kernel_size}, '
                f'wscale_gain={self.wscale_gain:.3f}, '
                f'bias={self.add_bias}, '
                f'lr_mul={self.lr_mul:.3f}, '
                f'upsample={self.scale_factor}, '
                f'upsample_filter={self.filter_kernel}, '
                f'act={self.activation_type}, '
                f'clamp={self.conv_clamp}')

    def forward(self, x, runtime_gain=1.0, impl='cuda'):
        dtype = x.dtype

        weight = self.weight
        if self.wscale != 1.0:
            weight = weight * self.wscale
        bias = None
        if self.bias is not None:
            bias = self.bias.to(dtype)
            if self.bscale != 1.0:
                bias = bias * self.bscale

        if self.scale_factor == 1:  # Native convolution without upsampling.
            padding = self.kernel_size // 2
            x = conv2d_gradfix.conv2d(
                x, weight.to(dtype), stride=1, padding=padding, impl=impl)
        else:  # Convolution with upsampling.
            up = self.scale_factor
            f = self.filter
            # When kernel size = 1, use filtering function for upsampling.
            if self.kernel_size == 1:
                padding = self.filter_padding
                x = conv2d_gradfix.conv2d(
                    x, weight.to(dtype), stride=1, padding=0, impl=impl)
                x = upfirdn2d.upfirdn2d(
                    x, f, up=up, padding=padding, gain=up ** 2, impl=impl)
            # When kernel size != 1, use transpose convolution for upsampling.
            else:
                # Following codes are borrowed from
                # https://github.com/NVlabs/stylegan2-ada-pytorch
                px0, px1, py0, py1 = self.filter_padding
                kh, kw = weight.shape[2:]
                px0 = px0 - (kw - 1)
                px1 = px1 - (kw - up)
                py0 = py0 - (kh - 1)
                py1 = py1 - (kh - up)
                pxt = max(min(-px0, -px1), 0)
                pyt = max(min(-py0, -py1), 0)
                weight = weight.transpose(0, 1)
                padding = (pyt, pxt)
                x = conv2d_gradfix.conv_transpose2d(
                    x, weight.to(dtype), stride=up, padding=padding, impl=impl)
                padding = (px0 + pxt, px1 + pxt, py0 + pyt, py1 + pyt)
                x = upfirdn2d.upfirdn2d(
                    x, f, up=1, padding=padding, gain=up ** 2, impl=impl)

        act_gain = self.act_gain * runtime_gain
        act_clamp = None
        if self.conv_clamp is not None:
            act_clamp = self.conv_clamp * runtime_gain
        x = bias_act.bias_act(x, bias,
                              act=self.activation_type,
                              gain=act_gain,
                              clamp=act_clamp,
                              impl=impl)

        assert x.dtype == dtype
        return x


class ModulateConvLayer(nn.Module):
    """Implements the convolutional layer with style modulation."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 resolution,
                 w_dim,
                 kernel_size,
                 add_bias,
                 scale_factor,
                 filter_kernel,
                 demodulate,
                 use_wscale,
                 wscale_gain,
                 lr_mul,
                 noise_type,
                 activation_type,
                 conv_clamp,
                 eps):
        """Initializes with layer settings.

        Args:
            in_channels: Number of channels of the input tensor.
            out_channels: Number of channels of the output tensor.
            resolution: Resolution of the output tensor.
            w_dim: Dimension of W space for style modulation.
            kernel_size: Size of the convolutional kernels.
            add_bias: Whether to add bias onto the convolutional result.
            scale_factor: Scale factor for upsampling.
            filter_kernel: Kernel used for filtering.
            demodulate: Whether to perform style demodulation.
            use_wscale: Whether to use weight scaling.
            wscale_gain: Gain factor for weight scaling.
            lr_mul: Learning multiplier for both weight and bias.
            noise_type: Type of noise added to the feature map after the
                convolution (if needed). Support `none`, `spatial` and
                `channel`.
            activation_type: Type of activation.
            conv_clamp: A threshold to clamp the output of convolution layers to
                avoid overflow under FP16 training.
            eps: A small value to avoid divide overflow.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.w_dim = w_dim
        self.kernel_size = kernel_size
        self.add_bias = add_bias
        self.scale_factor = scale_factor
        self.filter_kernel = filter_kernel
        self.demodulate = demodulate
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.lr_mul = lr_mul
        self.noise_type = noise_type.lower()
        self.activation_type = activation_type
        self.conv_clamp = conv_clamp
        self.eps = eps

        self.space_of_latent = 'W'

        # Set up weight.
        weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        fan_in = kernel_size * kernel_size * in_channels
        wscale = wscale_gain / np.sqrt(fan_in)
        if use_wscale:
            self.weight = nn.Parameter(torch.randn(*weight_shape) / lr_mul)
            self.wscale = wscale * lr_mul
        else:
            self.weight = nn.Parameter(
                torch.randn(*weight_shape) * wscale / lr_mul)
            self.wscale = lr_mul

        # Set up bias.
        if add_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
            self.bscale = lr_mul
        else:
            self.bias = None
        self.act_gain = bias_act.activation_funcs[activation_type].def_gain

        # Set up style.
        self.style = DenseLayer(in_channels=w_dim,
                                out_channels=in_channels,
                                add_bias=True,
                                init_bias=1.0,
                                use_wscale=use_wscale,
                                wscale_gain=wscale_gain,
                                lr_mul=lr_mul,
                                activation_type='linear')

        # Set up noise.
        if self.noise_type != 'none':
            self.noise_strength = nn.Parameter(torch.zeros(()))
            if self.noise_type == 'spatial':
                self.register_buffer(
                    'noise', torch.randn(1, 1, resolution, resolution))
            elif self.noise_type == 'channel':
                self.register_buffer(
                    'noise', torch.randn(1, out_channels, 1, 1))
            else:
                raise NotImplementedError(f'Not implemented noise type: '
                                          f'`{self.noise_type}`!')

        if scale_factor > 1:
            assert filter_kernel is not None
            self.register_buffer(
                'filter', upfirdn2d.setup_filter(filter_kernel))
            fh, fw = self.filter.shape
            self.filter_padding = (
                kernel_size // 2 + (fw + scale_factor - 1) // 2,
                kernel_size // 2 + (fw - scale_factor) // 2,
                kernel_size // 2 + (fh + scale_factor - 1) // 2,
                kernel_size // 2 + (fh - scale_factor) // 2)

    def extra_repr(self):
        return (f'in_ch={self.in_channels}, '
                f'out_ch={self.out_channels}, '
                f'ksize={self.kernel_size}, '
                f'wscale_gain={self.wscale_gain:.3f}, '
                f'bias={self.add_bias}, '
                f'lr_mul={self.lr_mul:.3f}, '
                f'upsample={self.scale_factor}, '
                f'upsample_filter={self.filter_kernel}, '
                f'demodulate={self.demodulate}, '
                f'noise_type={self.noise_type}, '
                f'act={self.activation_type}, '
                f'clamp={self.conv_clamp}')

    def forward_style(self, w, impl='cuda'):
        """Gets style code from the given input.

        More specifically, if the input is from W-Space, it will be projected by
        an affine transformation. If it is from the Style Space (Y-Space), no
        operation is required.

        NOTE: For codes from Y-Space, we use slicing to make sure the dimension
        is correct, in case that the code is padded before fed into this layer.
        """
        space_of_latent = self.space_of_latent.upper()
        if space_of_latent == 'W':
            if w.ndim != 2 or w.shape[1] != self.w_dim:
                raise ValueError(f'The input tensor should be with shape '
                                 f'[batch_size, w_dim], where '
                                 f'`w_dim` equals to {self.w_dim}!\n'
                                 f'But `{w.shape}` is received!')
            style = self.style(w, impl=impl)
        elif space_of_latent == 'Y':
            if w.ndim != 2 or w.shape[1] < self.in_channels:
                raise ValueError(f'The input tensor should be with shape '
                                 f'[batch_size, y_dim], where '
                                 f'`y_dim` equals to {self.in_channels}!\n'
                                 f'But `{w.shape}` is received!')
            style = w[:, :self.in_channels]
        else:
            raise NotImplementedError(f'Not implemented `space_of_latent`: '
                                      f'`{space_of_latent}`!')
        return style

    def forward(self,
                x,
                w,
                runtime_gain=1.0,
                noise_mode='const',
                fused_modulate=False,
                impl='cuda'):
        dtype = x.dtype
        N, C, H, W = x.shape

        fused_modulate = (fused_modulate and
                          not self.training and
                          (dtype == torch.float32 or N == 1))

        weight = self.weight
        out_ch, in_ch, kh, kw = weight.shape
        assert in_ch == C

        use_adain = self.w_dim != 0
        if use_adain:
            # Affine on `w`.
            style = self.forward_style(w, impl=impl)
            if not self.demodulate:
                _style = style * self.wscale  # Equivalent to scaling weight.
            else:
                _style = style

        # Prepare noise.
        noise = None
        noise_mode = noise_mode.lower()
        if self.noise_type != 'none' and noise_mode != 'none':
            if noise_mode == 'random':
                noise = torch.randn((N, *self.noise.shape[1:]), device=x.device)
            elif noise_mode == 'const':
                noise = self.noise
            else:
                raise ValueError(f'Unknown noise mode `{noise_mode}`!')
            noise = (noise * self.noise_strength).to(dtype)

        # Pre-normalize inputs to avoid FP16 overflow.
        if dtype == torch.float16 and self.demodulate:
            weight_max = weight.norm(float('inf'), dim=(1, 2, 3), keepdim=True)
            weight = weight * (self.wscale / weight_max)
            if use_adain:
                style_max = _style.norm(float('inf'), dim=1, keepdim=True)
                _style = _style / style_max

        if self.demodulate or fused_modulate:
            _weight = weight.unsqueeze(0)
            if use_adain:
                _weight = _weight * _style.reshape(N, 1, in_ch, 1, 1)
            else:
                _weight = _weight.repeat(N, 1, 1, 1, 1)
        if self.demodulate:
            decoef = (_weight.square().sum(dim=(2, 3, 4)) + self.eps).rsqrt()
        if self.demodulate and fused_modulate:
            _weight = _weight * decoef.reshape(N, out_ch, 1, 1, 1)

        if not fused_modulate:
            if use_adain:
                x = x * _style.to(dtype).reshape(N, in_ch, 1, 1)
            w = weight.to(dtype)
            groups = 1
        else:  # Use group convolution to fuse style modulation and convolution.
            x = x.reshape(1, N * in_ch, H, W)
            w = _weight.reshape(N * out_ch, in_ch, kh, kw).to(dtype)
            groups = N

        if self.scale_factor == 1:  # Native convolution without upsampling.
            up = 1
            padding = self.kernel_size // 2
            x = conv2d_gradfix.conv2d(
                x, w, stride=1, padding=padding, groups=groups, impl=impl)
        else:  # Convolution with upsampling.
            up = self.scale_factor
            f = self.filter
            # When kernel size = 1, use filtering function for upsampling.
            if self.kernel_size == 1:
                padding = self.filter_padding
                x = conv2d_gradfix.conv2d(
                    x, w, stride=1, padding=0, groups=groups, impl=impl)
                x = upfirdn2d.upfirdn2d(
                    x, f, up=up, padding=padding, gain=up ** 2, impl=impl)
            # When kernel size != 1, use stride convolution for upsampling.
            else:
                # Following codes are borrowed from
                # https://github.com/NVlabs/stylegan2-ada-pytorch
                px0, px1, py0, py1 = self.filter_padding
                px0 = px0 - (kw - 1)
                px1 = px1 - (kw - up)
                py0 = py0 - (kh - 1)
                py1 = py1 - (kh - up)
                pxt = max(min(-px0, -px1), 0)
                pyt = max(min(-py0, -py1), 0)
                if groups == 1:
                    w = w.transpose(0, 1)
                else:
                    w = w.reshape(N, out_ch, in_ch, kh, kw)
                    w = w.transpose(1, 2)
                    w = w.reshape(N * in_ch, out_ch, kh, kw)
                padding = (pyt, pxt)
                x = conv2d_gradfix.conv_transpose2d(
                    x, w, stride=up, padding=padding, groups=groups, impl=impl)
                padding = (px0 + pxt, px1 + pxt, py0 + pyt, py1 + pyt)
                x = upfirdn2d.upfirdn2d(
                    x, f, up=1, padding=padding, gain=up ** 2, impl=impl)

        if not fused_modulate:
            if self.demodulate:
                decoef = decoef.to(dtype).reshape(N, out_ch, 1, 1)
            if self.demodulate and noise is not None:
                x = fma.fma(x, decoef, noise, impl=impl)
            else:
                if self.demodulate:
                    x = x * decoef
                if noise is not None:
                    x = x + noise
        else:
            x = x.reshape(N, out_ch, H * up, W * up)
            if noise is not None:
                x = x + noise

        bias = None
        if self.bias is not None:
            bias = self.bias.to(dtype)
            if self.bscale != 1.0:
                bias = bias * self.bscale

        if self.activation_type == 'linear':  # Shortcut for output layer.
            x = bias_act.bias_act(
                x, bias, act='linear', clamp=self.conv_clamp, impl=impl)
        else:
            act_gain = self.act_gain * runtime_gain
            act_clamp = None
            if self.conv_clamp is not None:
                act_clamp = self.conv_clamp * runtime_gain
            x = bias_act.bias_act(x, bias,
                                  act=self.activation_type,
                                  gain=act_gain,
                                  clamp=act_clamp,
                                  impl=impl)

        assert x.dtype == dtype
        if use_adain:
            assert style.dtype == torch.float32
            return x, style
        else:
            return x


class DenseLayer(nn.Module):
    """Implements the dense layer."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 add_bias,
                 init_bias,
                 use_wscale,
                 wscale_gain,
                 lr_mul,
                 activation_type):
        """Initializes with layer settings.

        Args:
            in_channels: Number of channels of the input tensor.
            out_channels: Number of channels of the output tensor.
            add_bias: Whether to add bias onto the fully-connected result.
            init_bias: The initial bias value before training.
            use_wscale: Whether to use weight scaling.
            wscale_gain: Gain factor for weight scaling.
            lr_mul: Learning multiplier for both weight and bias.
            activation_type: Type of activation.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_bias = add_bias
        self.init_bias = init_bias
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.lr_mul = lr_mul
        self.activation_type = activation_type

        weight_shape = (out_channels, in_channels)
        wscale = wscale_gain / np.sqrt(in_channels)
        if use_wscale:
            self.weight = nn.Parameter(torch.randn(*weight_shape) / lr_mul)
            self.wscale = wscale * lr_mul
        else:
            self.weight = nn.Parameter(
                torch.randn(*weight_shape) * wscale / lr_mul)
            self.wscale = lr_mul

        if add_bias:
            init_bias = np.float32(init_bias) / lr_mul
            self.bias = nn.Parameter(torch.full([out_channels], init_bias))
            self.bscale = lr_mul
        else:
            self.bias = None

    def extra_repr(self):
        return (f'in_ch={self.in_channels}, '
                f'out_ch={self.out_channels}, '
                f'wscale_gain={self.wscale_gain:.3f}, '
                f'bias={self.add_bias}, '
                f'init_bias={self.init_bias}, '
                f'lr_mul={self.lr_mul:.3f}, '
                f'act={self.activation_type}')

    def forward(self, x, impl='cuda'):
        dtype = x.dtype

        if x.ndim != 2:
            x = x.flatten(start_dim=1)

        weight = self.weight.to(dtype) * self.wscale
        bias = None
        if self.bias is not None:
            bias = self.bias.to(dtype)
            if self.bscale != 1.0:
                bias = bias * self.bscale

        # Fast pass for linear activation.
        if self.activation_type == 'linear' and bias is not None:
            x = torch.addmm(bias.unsqueeze(0), x, weight.t())
        elif self.activation_type == 'softplus':
            x = torch.addmm(bias.unsqueeze(0), x, weight.t())
            x = F.softplus(x)
        else:
            x = x.matmul(weight.t())
            x = bias_act.bias_act(x, bias, act=self.activation_type, impl=impl)

        assert x.dtype == dtype
        return x

# pylint: enable=missing-function-docstring
