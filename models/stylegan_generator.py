# python3.7
"""Contains the implementation of generator described in StyleGAN.

Paper: https://arxiv.org/pdf/1812.04948.pdf

Official TensorFlow implementation: https://github.com/NVlabs/stylegan
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from .utils.ops import all_gather

__all__ = ['StyleGANGenerator']

# Resolutions allowed.
_RESOLUTIONS_ALLOWED = [8, 16, 32, 64, 128, 256, 512, 1024]

# Fused-scale options allowed.
_FUSED_SCALE_ALLOWED = [True, False, 'auto']

# pylint: disable=missing-function-docstring

class StyleGANGenerator(nn.Module):
    """Defines the generator network in StyleGAN.

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
        mapping network (default: sqrt(2.0))
    (9) mapping_lr_mul: Learning rate multiplier for the mapping network.
        (default: 0.01)

    Settings for conditional generation:

    (1) label_dim: Dimension of the additional label for conditional generation.
        In one-hot conditioning case, it is equal to the number of classes. If
        set to 0, conditioning training will be disabled. (default: 0)
    (2) embedding_dim: Dimension of the embedding space, if needed.
        (default: 512)

    Settings for the synthesis network:

    (1) resolution: The resolution of the output image. (default: -1)
    (2) init_res: The initial resolution to start with convolution. (default: 4)
    (3) image_channels: Number of channels of the output image. (default: 3)
    (4) final_tanh: Whether to use `tanh` to control the final pixel range.
        (default: False)
    (5) fused_scale:  The strategy of fusing `upsample` and `conv2d` as one
        operator. `True` means blocks from all resolutions will fuse. `False`
        means blocks from all resolutions will not fuse. `auto` means blocks
        from resolutions higher than (or equal to) `fused_scale_res` will fuse.
        (default: `auto`)
    (6) fused_scale_res: Minimum resolution to fuse `conv2d` and `downsample`
        as one operator. This field only takes effect if `fused_scale` is set
        as `auto`. (default: 128)
    (7) use_wscale: Whether to use weight scaling. (default: True)
    (8) wscale_gain: The factor to control weight scaling. (default: sqrt(2.0))
    (9) lr_mul: Learning rate multiplier for the synthesis network.
        (default: 1.0)
    (10) noise_type: Type of noise added to the convolutional results at each
         layer. (default: `spatial`)
    (11) fmaps_base: Factor to control number of feature maps for each layer.
         (default: 16 << 10)
    (12) fmaps_max: Maximum number of feature maps in each layer. (default: 512)
    (13) filter_kernel: Kernel used for filtering (e.g., downsampling).
         (default: (1, 2, 1))
    (14) eps: A small value to avoid divide overflow. (default: 1e-8)

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
    (7) enable_amp: Whether to enable automatic mixed precision training.
        (default: False)
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
                 mapping_wscale_gain=np.sqrt(2.0),
                 mapping_lr_mul=0.01,
                 # Settings for conditional generation.
                 label_dim=0,
                 embedding_dim=512,
                 # Settings for synthesis network.
                 resolution=-1,
                 init_res=4,
                 image_channels=3,
                 final_tanh=False,
                 fused_scale='auto',
                 fused_scale_res=128,
                 use_wscale=True,
                 wscale_gain=np.sqrt(2.0),
                 lr_mul=1.0,
                 noise_type='spatial',
                 fmaps_base=16 << 10,
                 fmaps_max=512,
                 filter_kernel=(1, 2, 1),
                 eps=1e-8):
        """Initializes with basic settings.

        Raises:
            ValueError: If the `resolution` is not supported, or `fused_scale`
                is not supported.
        """
        super().__init__()

        if resolution not in _RESOLUTIONS_ALLOWED:
            raise ValueError(f'Invalid resolution: `{resolution}`!\n'
                             f'Resolutions allowed: {_RESOLUTIONS_ALLOWED}.')
        if fused_scale not in _FUSED_SCALE_ALLOWED:
            raise ValueError(f'Invalid fused-scale option: `{fused_scale}`!\n'
                             f'Options allowed: {_FUSED_SCALE_ALLOWED}.')

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

        self.resolution = resolution
        self.init_res = init_res
        self.image_channels = image_channels
        self.final_tanh = final_tanh
        self.fused_scale = fused_scale
        self.fused_scale_res = fused_scale_res
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.lr_mul = lr_mul
        self.noise_type = noise_type.lower()
        self.fmaps_base = fmaps_base
        self.fmaps_max = fmaps_max
        self.filter_kernel = filter_kernel
        self.eps = eps

        # Dimension of latent space, which is convenient for sampling.
        self.latent_dim = (z_dim,)

        # Number of synthesis (convolutional) layers.
        self.num_layers = int(np.log2(resolution // init_res * 2)) * 2

        self.mapping = MappingNetwork(input_dim=z_dim,
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
                                          fused_scale=fused_scale,
                                          fused_scale_res=fused_scale_res,
                                          use_wscale=use_wscale,
                                          wscale_gain=wscale_gain,
                                          lr_mul=lr_mul,
                                          noise_type=noise_type,
                                          fmaps_base=fmaps_base,
                                          fmaps_max=fmaps_max,
                                          filter_kernel=filter_kernel,
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
                lod=None,
                w_moving_decay=None,
                sync_w_avg=False,
                style_mixing_prob=None,
                trunc_psi=None,
                trunc_layers=None,
                noise_mode='const',
                enable_amp=False):
        mapping_results = self.mapping(z, label)

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
                new_wp = self.mapping(new_z, label)['wp']
                lod = self.synthesis.lod.item() if lod is None else lod
                current_layers = self.num_layers - int(lod) * 2
                mixing_cutoff = np.random.randint(1, current_layers)
                wp[:, mixing_cutoff:] = new_wp[:, mixing_cutoff:]

        if not self.training:
            trunc_psi = 1.0 if trunc_psi is None else trunc_psi
            trunc_layers = 0 if trunc_layers is None else trunc_layers
            if trunc_psi < 1.0 and trunc_layers > 0:
                w_avg = self.w_avg.reshape(1, -1, self.w_dim)[:, :trunc_layers]
                wp[:, :trunc_layers] = w_avg.lerp(
                    wp[:, :trunc_layers], trunc_psi)

        with autocast(enabled=enable_amp):
            synthesis_results = self.synthesis(wp,
                                               lod=lod,
                                               noise_mode=noise_mode)

        return {**mapping_results, **synthesis_results}


class MappingNetwork(nn.Module):
    """Implements the latent space mapping module.

    Basically, this module executes several dense layers in sequence, and the
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
        self.eps = eps

        self.pth_to_tf_var_mapping = {}

        if normalize_input:
            self.norm = PixelNormLayer(dim=1, eps=eps)

        if self.label_dim > 0:
            input_dim = input_dim + embedding_dim
            self.embedding = nn.Parameter(
                torch.randn(label_dim, embedding_dim))
            self.pth_to_tf_var_mapping['embedding'] = 'LabelConcat/weight'

        if num_outputs is not None and not repeat_output:
            output_dim = output_dim * num_outputs
        for i in range(num_layers):
            in_channels = (input_dim if i == 0 else hidden_dim)
            out_channels = (output_dim if i == (num_layers - 1) else hidden_dim)
            self.add_module(f'dense{i}',
                            DenseLayer(in_channels=in_channels,
                                       out_channels=out_channels,
                                       add_bias=True,
                                       use_wscale=use_wscale,
                                       wscale_gain=wscale_gain,
                                       lr_mul=lr_mul,
                                       activation_type='lrelu'))
            self.pth_to_tf_var_mapping[f'dense{i}.weight'] = f'Dense{i}/weight'
            self.pth_to_tf_var_mapping[f'dense{i}.bias'] = f'Dense{i}/bias'

    def forward(self, z, label=None):
        if z.ndim != 2 or z.shape[1] != self.input_dim:
            raise ValueError(f'Input latent code should be with shape '
                             f'[batch_size, input_dim], where '
                             f'`input_dim` equals to {self.input_dim}!\n'
                             f'But `{z.shape}` is received!')

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
            embedding = torch.matmul(label, self.embedding)
            z = torch.cat((z, embedding), dim=1)

        if self.normalize_input:
            w = self.norm(z)
        else:
            w = z

        for i in range(self.num_layers):
            w = getattr(self, f'dense{i}')(w)

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
    """Implements the image synthesis module.

    Basically, this module executes several convolutional layers in sequence.
    """

    def __init__(self,
                 resolution,
                 init_res,
                 w_dim,
                 image_channels,
                 final_tanh,
                 fused_scale,
                 fused_scale_res,
                 use_wscale,
                 wscale_gain,
                 lr_mul,
                 noise_type,
                 fmaps_base,
                 fmaps_max,
                 filter_kernel,
                 eps):
        super().__init__()

        self.init_res = init_res
        self.init_res_log2 = int(np.log2(init_res))
        self.resolution = resolution
        self.final_res_log2 = int(np.log2(resolution))
        self.w_dim = w_dim
        self.image_channels = image_channels
        self.final_tanh = final_tanh
        self.fused_scale = fused_scale
        self.fused_scale_res = fused_scale_res
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.lr_mul = lr_mul
        self.noise_type = noise_type.lower()
        self.fmaps_base = fmaps_base
        self.fmaps_max = fmaps_max
        self.eps = eps

        self.num_layers = (self.final_res_log2 - self.init_res_log2 + 1) * 2

        # Level-of-details (used for progressive training).
        self.register_buffer('lod', torch.zeros(()))
        self.pth_to_tf_var_mapping = {'lod': 'lod'}

        for res_log2 in range(self.init_res_log2, self.final_res_log2 + 1):
            res = 2 ** res_log2
            in_channels = self.get_nf(res // 2)
            out_channels = self.get_nf(res)
            block_idx = res_log2 - self.init_res_log2

            # First layer (kernel 3x3) with upsampling
            layer_name = f'layer{2 * block_idx}'
            if res == self.init_res:
                self.add_module(layer_name,
                                ModulateConvLayer(in_channels=0,
                                                  out_channels=out_channels,
                                                  resolution=res,
                                                  w_dim=w_dim,
                                                  kernel_size=None,
                                                  add_bias=True,
                                                  scale_factor=None,
                                                  fused_scale=None,
                                                  filter_kernel=None,
                                                  use_wscale=use_wscale,
                                                  wscale_gain=wscale_gain,
                                                  lr_mul=lr_mul,
                                                  noise_type=noise_type,
                                                  activation_type='lrelu',
                                                  use_style=True,
                                                  eps=eps))
                tf_layer_name = 'Const'
                self.pth_to_tf_var_mapping[f'{layer_name}.const'] = (
                    f'{res}x{res}/{tf_layer_name}/const')
            else:
                self.add_module(
                    layer_name,
                    ModulateConvLayer(in_channels=in_channels,
                                      out_channels=out_channels,
                                      resolution=res,
                                      w_dim=w_dim,
                                      kernel_size=3,
                                      add_bias=True,
                                      scale_factor=2,
                                      fused_scale=(res >= fused_scale_res
                                                   if fused_scale == 'auto'
                                                   else fused_scale),
                                      filter_kernel=filter_kernel,
                                      use_wscale=use_wscale,
                                      wscale_gain=wscale_gain,
                                      lr_mul=lr_mul,
                                      noise_type=noise_type,
                                      activation_type='lrelu',
                                      use_style=True,
                                      eps=eps))
                tf_layer_name = 'Conv0_up'
                self.pth_to_tf_var_mapping[f'{layer_name}.weight'] = (
                    f'{res}x{res}/{tf_layer_name}/weight')
            self.pth_to_tf_var_mapping[f'{layer_name}.bias'] = (
                f'{res}x{res}/{tf_layer_name}/bias')
            self.pth_to_tf_var_mapping[f'{layer_name}.style.weight'] = (
                f'{res}x{res}/{tf_layer_name}/StyleMod/weight')
            self.pth_to_tf_var_mapping[f'{layer_name}.style.bias'] = (
                f'{res}x{res}/{tf_layer_name}/StyleMod/bias')
            self.pth_to_tf_var_mapping[f'{layer_name}.noise_strength'] = (
                f'{res}x{res}/{tf_layer_name}/Noise/weight')
            self.pth_to_tf_var_mapping[f'{layer_name}.noise'] = (
                f'noise{2 * block_idx}')

            # Second layer (kernel 3x3) without upsampling.
            layer_name = f'layer{2 * block_idx + 1}'
            self.add_module(layer_name,
                            ModulateConvLayer(in_channels=out_channels,
                                              out_channels=out_channels,
                                              resolution=res,
                                              w_dim=w_dim,
                                              kernel_size=3,
                                              add_bias=True,
                                              scale_factor=1,
                                              fused_scale=False,
                                              filter_kernel=None,
                                              use_wscale=use_wscale,
                                              wscale_gain=wscale_gain,
                                              lr_mul=lr_mul,
                                              noise_type=noise_type,
                                              activation_type='lrelu',
                                              use_style=True,
                                              eps=eps))
            tf_layer_name = 'Conv' if res == self.init_res else 'Conv1'
            self.pth_to_tf_var_mapping[f'{layer_name}.weight'] = (
                f'{res}x{res}/{tf_layer_name}/weight')
            self.pth_to_tf_var_mapping[f'{layer_name}.bias'] = (
                f'{res}x{res}/{tf_layer_name}/bias')
            self.pth_to_tf_var_mapping[f'{layer_name}.style.weight'] = (
                f'{res}x{res}/{tf_layer_name}/StyleMod/weight')
            self.pth_to_tf_var_mapping[f'{layer_name}.style.bias'] = (
                f'{res}x{res}/{tf_layer_name}/StyleMod/bias')
            self.pth_to_tf_var_mapping[f'{layer_name}.noise_strength'] = (
                f'{res}x{res}/{tf_layer_name}/Noise/weight')
            self.pth_to_tf_var_mapping[f'{layer_name}.noise'] = (
                f'noise{2 * block_idx + 1}')

            # Output convolution layer for each resolution.
            self.add_module(f'output{block_idx}',
                            ModulateConvLayer(in_channels=out_channels,
                                              out_channels=image_channels,
                                              resolution=res,
                                              w_dim=w_dim,
                                              kernel_size=1,
                                              add_bias=True,
                                              scale_factor=1,
                                              fused_scale=False,
                                              filter_kernel=None,
                                              use_wscale=use_wscale,
                                              wscale_gain=1.0,
                                              lr_mul=lr_mul,
                                              noise_type='none',
                                              activation_type='linear',
                                              use_style=False,
                                              eps=eps))
            self.pth_to_tf_var_mapping[f'output{block_idx}.weight'] = (
                f'ToRGB_lod{self.final_res_log2 - res_log2}/weight')
            self.pth_to_tf_var_mapping[f'output{block_idx}.bias'] = (
                f'ToRGB_lod{self.final_res_log2 - res_log2}/bias')

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
            if isinstance(module, ModulateConvLayer) and module.use_style:
                setattr(module, 'space_of_latent', space_of_latent)

    def forward(self, wp, lod=None, noise_mode='const'):
        lod = self.lod.item() if lod is None else lod
        if lod + self.init_res_log2 > self.final_res_log2:
            raise ValueError(f'Maximum level-of-details (lod) is '
                             f'{self.final_res_log2 - self.init_res_log2}, '
                             f'but `{lod}` is received!')

        results = {'wp': wp}
        x = None
        for res_log2 in range(self.init_res_log2, self.final_res_log2 + 1):
            current_lod = self.final_res_log2 - res_log2
            block_idx = res_log2 - self.init_res_log2
            if lod < current_lod + 1:
                layer = getattr(self, f'layer{2 * block_idx}')
                x, style = layer(x, wp[:, 2 * block_idx], noise_mode)
                results[f'style{2 * block_idx}'] = style
                layer = getattr(self, f'layer{2 * block_idx + 1}')
                x, style = layer(x, wp[:, 2 * block_idx + 1], noise_mode)
                results[f'style{2 * block_idx + 1}'] = style
            if current_lod - 1 < lod <= current_lod:
                image = getattr(self, f'output{block_idx}')(x)
            elif current_lod < lod < current_lod + 1:
                alpha = np.ceil(lod) - lod
                temp = getattr(self, f'output{block_idx}')(x)
                image = F.interpolate(image, scale_factor=2, mode='nearest')
                image = temp * alpha + image * (1 - alpha)
            elif lod >= current_lod + 1:
                image = F.interpolate(image, scale_factor=2, mode='nearest')

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


class Blur(torch.autograd.Function):
    """Defines blur operation with customized gradient computation."""

    @staticmethod
    def forward(ctx, x, kernel):
        assert kernel.shape[2] == 3 and kernel.shape[3] == 3
        ctx.save_for_backward(kernel)
        y = F.conv2d(input=x,
                     weight=kernel,
                     bias=None,
                     stride=1,
                     padding=1,
                     groups=x.shape[1])
        return y

    @staticmethod
    def backward(ctx, dy):
        kernel, = ctx.saved_tensors
        dx = F.conv2d(input=dy,
                      weight=kernel.flip((2, 3)),
                      bias=None,
                      stride=1,
                      padding=1,
                      groups=dy.shape[1])
        return dx, None, None


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
                 fused_scale,
                 filter_kernel,
                 use_wscale,
                 wscale_gain,
                 lr_mul,
                 noise_type,
                 activation_type,
                 use_style,
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
            fused_scale: Whether to fuse `upsample` and `conv2d` as one
                operator, using transpose convolution.
            filter_kernel: Kernel used for filtering.
            use_wscale: Whether to use weight scaling.
            wscale_gain: Gain factor for weight scaling.
            lr_mul: Learning multiplier for both weight and bias.
            noise_type: Type of noise added to the feature map after the
                convolution (if needed). Support `none`, `spatial` and
                `channel`.
            activation_type: Type of activation.
            use_style: Whether to apply style modulation.
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
        self.fused_scale = fused_scale
        self.filter_kernel = filter_kernel
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.lr_mul = lr_mul
        self.noise_type = noise_type.lower()
        self.activation_type = activation_type
        self.use_style = use_style
        self.eps = eps

        # Set up noise.
        if self.noise_type == 'none':
            pass
        elif self.noise_type == 'spatial':
            self.register_buffer(
                'noise', torch.randn(1, 1, resolution, resolution))
            self.noise_strength = nn.Parameter(
                torch.zeros(1, out_channels, 1, 1))
        elif self.noise_type == 'channel':
            self.register_buffer(
                'noise', torch.randn(1, out_channels, 1, 1))
            self.noise_strength = nn.Parameter(
                torch.zeros(1, 1, resolution, resolution))
        else:
            raise NotImplementedError(f'Not implemented noise type: '
                                      f'`{noise_type}`!')

        # Set up bias.
        if add_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
            self.bscale = lr_mul
        else:
            self.bias = None

        # Set up activation.
        assert activation_type in ['linear', 'relu', 'lrelu']

        # Set up style.
        if use_style:
            self.space_of_latent = 'W'
            self.style = DenseLayer(in_channels=w_dim,
                                    out_channels=out_channels * 2,
                                    add_bias=True,
                                    use_wscale=use_wscale,
                                    wscale_gain=1.0,
                                    lr_mul=1.0,
                                    activation_type='linear')

        if in_channels == 0:  # First layer.
            self.const = nn.Parameter(
                torch.ones(1, out_channels, resolution, resolution))
            return

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

        # Set up upsampling filter (if needed).
        if scale_factor > 1:
            assert filter_kernel is not None
            kernel = np.array(filter_kernel, dtype=np.float32).reshape(1, -1)
            kernel = kernel.T.dot(kernel)
            kernel = kernel / np.sum(kernel)
            kernel = kernel[np.newaxis, np.newaxis]
            self.register_buffer('filter', torch.from_numpy(kernel))

        if scale_factor > 1 and fused_scale:  # use transpose convolution.
            self.stride = scale_factor
        else:
            self.stride = 1
        self.padding = kernel_size // 2

    def extra_repr(self):
        return (f'in_ch={self.in_channels}, '
                f'out_ch={self.out_channels}, '
                f'ksize={self.kernel_size}, '
                f'wscale_gain={self.wscale_gain:.3f}, '
                f'bias={self.add_bias}, '
                f'lr_mul={self.lr_mul:.3f}, '
                f'upsample={self.scale_factor}, '
                f'fused_scale={self.fused_scale}, '
                f'upsample_filter={self.filter_kernel}, '
                f'noise_type={self.noise_type}, '
                f'act={self.activation_type}, '
                f'use_style={self.use_style}')

    def forward_style(self, w):
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
            style = self.style(w)
        elif space_of_latent == 'Y':
            if w.ndim != 2 or w.shape[1] < self.out_channels * 2:
                raise ValueError(f'The input tensor should be with shape '
                                 f'[batch_size, y_dim], where '
                                 f'`y_dim` equals to {self.out_channels * 2}!\n'
                                 f'But `{w.shape}` is received!')
            style = w[:, :self.out_channels * 2]
        else:
            raise NotImplementedError(f'Not implemented `space_of_latent`: '
                                      f'`{space_of_latent}`!')
        return style

    def forward(self, x, w=None, noise_mode='const'):
        if self.in_channels == 0:
            assert x is None
            x = self.const.repeat(w.shape[0], 1, 1, 1)
        else:
            weight = self.weight
            if self.wscale != 1.0:
                weight = weight * self.wscale

            if self.scale_factor > 1 and self.fused_scale:
                weight = F.pad(weight, (1, 1, 1, 1, 0, 0, 0, 0), 'constant', 0)
                weight = (weight[:, :, 1:, 1:] + weight[:, :, :-1, 1:] +
                          weight[:, :, 1:, :-1] + weight[:, :, :-1, :-1])
                x = F.conv_transpose2d(x,
                                       weight=weight.transpose(0, 1),
                                       bias=None,
                                       stride=self.stride,
                                       padding=self.padding)
            else:
                if self.scale_factor > 1:
                    up = self.scale_factor
                    x = F.interpolate(x, scale_factor=up, mode='nearest')
                x = F.conv2d(x,
                             weight=weight,
                             bias=None,
                             stride=self.stride,
                             padding=self.padding)

            if self.scale_factor > 1:
                # Disable `autocast` for customized autograd function.
                # Please check reference:
                # https://pytorch.org/docs/stable/notes/amp_examples.html#autocast-and-custom-autograd-functions
                with autocast(enabled=False):
                    f = self.filter.repeat(self.out_channels, 1, 1, 1)
                    x = Blur.apply(x.float(), f)  # Always use FP32.

        # Prepare noise.
        noise_mode = noise_mode.lower()
        if self.noise_type != 'none' and noise_mode != 'none':
            if noise_mode == 'random':
                noise = torch.randn(
                    (x.shape[0], *self.noise.shape[1:]), device=x.device)
            elif noise_mode == 'const':
                noise = self.noise
            else:
                raise ValueError(f'Unknown noise mode `{noise_mode}`!')
            x = x + noise * self.noise_strength

        if self.bias is not None:
            bias = self.bias
            if self.bscale != 1.0:
                bias = bias * self.bscale
            x = x + bias.reshape(1, self.out_channels, 1, 1)

        if self.activation_type == 'linear':
            pass
        elif self.activation_type == 'relu':
            x = F.relu(x, inplace=True)
        elif self.activation_type == 'lrelu':
            x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        else:
            raise NotImplementedError(f'Not implemented activation type '
                                      f'`{self.activation_type}`!')

        if not self.use_style:
            return x

        # Instance normalization.
        x = x - x.mean(dim=(2, 3), keepdim=True)
        scale = (x.square().mean(dim=(2, 3), keepdim=True) + self.eps).rsqrt()
        x = x * scale
        # Style modulation.
        style = self.forward_style(w)
        style_split = style.unsqueeze(2).unsqueeze(3).chunk(2, dim=1)
        x = x * (style_split[0] + 1) + style_split[1]

        return x, style


class DenseLayer(nn.Module):
    """Implements the dense layer."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 add_bias,
                 use_wscale,
                 wscale_gain,
                 lr_mul,
                 activation_type):
        """Initializes with layer settings.

        Args:
            in_channels: Number of channels of the input tensor.
            out_channels: Number of channels of the output tensor.
            add_bias: Whether to add bias onto the fully-connected result.
            use_wscale: Whether to use weight scaling.
            wscale_gain: Gain factor for weight scaling.
            lr_mul: Learning multiplier for both weight and bias.
            activation_type: Type of activation.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_bias = add_bias
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
            self.bias = nn.Parameter(torch.zeros(out_channels))
            self.bscale = lr_mul
        else:
            self.bias = None

        assert activation_type in ['linear', 'relu', 'lrelu']

    def extra_repr(self):
        return (f'in_ch={self.in_channels}, '
                f'out_ch={self.out_channels}, '
                f'wscale_gain={self.wscale_gain:.3f}, '
                f'bias={self.add_bias}, '
                f'lr_mul={self.lr_mul:.3f}, '
                f'act={self.activation_type}')

    def forward(self, x):
        if x.ndim != 2:
            x = x.flatten(start_dim=1)

        weight = self.weight
        if self.wscale != 1.0:
            weight = weight * self.wscale
        bias = None
        if self.bias is not None:
            bias = self.bias
            if self.bscale != 1.0:
                bias = bias * self.bscale

        x = F.linear(x, weight=weight, bias=bias)

        if self.activation_type == 'linear':
            pass
        elif self.activation_type == 'relu':
            x = F.relu(x, inplace=True)
        elif self.activation_type == 'lrelu':
            x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        else:
            raise NotImplementedError(f'Not implemented activation type '
                                      f'`{self.activation_type}`!')

        return x

# pylint: enable=missing-function-docstring
