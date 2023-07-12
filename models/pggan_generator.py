# python3.7
"""Contains the implementation of generator described in PGGAN.

Paper: https://arxiv.org/pdf/1710.10196.pdf

Official TensorFlow implementation:
https://github.com/tkarras/progressive_growing_of_gans
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['PGGANGenerator']

# Resolutions allowed.
_RESOLUTIONS_ALLOWED = [8, 16, 32, 64, 128, 256, 512, 1024]

# pylint: disable=missing-function-docstring

class PGGANGenerator(nn.Module):
    """Defines the generator network in PGGAN.

    NOTE: The synthesized images are with `RGB` channel order and pixel range
    [-1, 1].

    Settings for the network:

    (1) resolution: The resolution of the output image.
    (2) init_res: The initial resolution to start with convolution. (default: 4)
    (3) z_dim: Dimension of the input latent space, Z. (default: 512)
    (4) image_channels: Number of channels of the output image. (default: 3)
    (5) final_tanh: Whether to use `tanh` to control the final pixel range.
        (default: False)
    (6) label_dim: Dimension of the additional label for conditional generation.
        In one-hot conditioning case, it is equal to the number of classes. If
        set to 0, conditioning training will be disabled. (default: 0)
    (7) fused_scale: Whether to fused `upsample` and `conv2d` together,
        resulting in `conv2d_transpose`. (default: False)
    (8) use_wscale: Whether to use weight scaling. (default: True)
    (9) wscale_gain: The factor to control weight scaling. (default: sqrt(2.0))
    (10) fmaps_base: Factor to control number of feature maps for each layer.
         (default: 16 << 10)
    (11) fmaps_max: Maximum number of feature maps in each layer. (default: 512)
    (12) eps: A small value to avoid divide overflow. (default: 1e-8)
    """

    def __init__(self,
                 resolution,
                 init_res=4,
                 z_dim=512,
                 image_channels=3,
                 final_tanh=False,
                 label_dim=0,
                 fused_scale=False,
                 use_wscale=True,
                 wscale_gain=np.sqrt(2.0),
                 fmaps_base=16 << 10,
                 fmaps_max=512,
                 eps=1e-8):
        """Initializes with basic settings.

        Raises:
            ValueError: If the `resolution` is not supported.
        """
        super().__init__()

        if resolution not in _RESOLUTIONS_ALLOWED:
            raise ValueError(f'Invalid resolution: `{resolution}`!\n'
                             f'Resolutions allowed: {_RESOLUTIONS_ALLOWED}.')

        self.init_res = init_res
        self.init_res_log2 = int(np.log2(self.init_res))
        self.resolution = resolution
        self.final_res_log2 = int(np.log2(self.resolution))
        self.z_dim = z_dim
        self.image_channels = image_channels
        self.final_tanh = final_tanh
        self.label_dim = label_dim
        self.fused_scale = fused_scale
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.fmaps_base = fmaps_base
        self.fmaps_max = fmaps_max
        self.eps = eps

        # Dimension of latent space, which is convenient for sampling.
        self.latent_dim = (self.z_dim,)

        # Number of convolutional layers.
        self.num_layers = (self.final_res_log2 - self.init_res_log2 + 1) * 2

        # Level-of-details (used for progressive training).
        self.register_buffer('lod', torch.zeros(()))
        self.pth_to_tf_var_mapping = {'lod': 'lod'}

        for res_log2 in range(self.init_res_log2, self.final_res_log2 + 1):
            res = 2 ** res_log2
            in_channels = self.get_nf(res // 2)
            out_channels = self.get_nf(res)
            block_idx = res_log2 - self.init_res_log2

            # First convolution layer for each resolution.
            if res == self.init_res:
                self.add_module(
                    f'layer{2 * block_idx}',
                    ConvLayer(in_channels=z_dim + label_dim,
                              out_channels=out_channels,
                              kernel_size=init_res,
                              padding=init_res - 1,
                              add_bias=True,
                              upsample=False,
                              fused_scale=False,
                              use_wscale=use_wscale,
                              wscale_gain=wscale_gain,
                              activation_type='lrelu',
                              eps=eps))
                tf_layer_name = 'Dense'
            else:
                self.add_module(
                    f'layer{2 * block_idx}',
                    ConvLayer(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              padding=1,
                              add_bias=True,
                              upsample=True,
                              fused_scale=fused_scale,
                              use_wscale=use_wscale,
                              wscale_gain=wscale_gain,
                              activation_type='lrelu',
                              eps=eps))
                tf_layer_name = 'Conv0_up' if fused_scale else 'Conv0'
            self.pth_to_tf_var_mapping[f'layer{2 * block_idx}.weight'] = (
                f'{res}x{res}/{tf_layer_name}/weight')
            self.pth_to_tf_var_mapping[f'layer{2 * block_idx}.bias'] = (
                f'{res}x{res}/{tf_layer_name}/bias')

            # Second convolution layer for each resolution.
            self.add_module(
                f'layer{2 * block_idx + 1}',
                ConvLayer(in_channels=out_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          padding=1,
                          add_bias=True,
                          upsample=False,
                          fused_scale=False,
                          use_wscale=use_wscale,
                          wscale_gain=wscale_gain,
                          activation_type='lrelu',
                          eps=eps))
            tf_layer_name = 'Conv' if res == self.init_res else 'Conv1'
            self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 1}.weight'] = (
                f'{res}x{res}/{tf_layer_name}/weight')
            self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 1}.bias'] = (
                f'{res}x{res}/{tf_layer_name}/bias')

            # Output convolution layer for each resolution.
            self.add_module(
                f'output{block_idx}',
                ConvLayer(in_channels=out_channels,
                          out_channels=image_channels,
                          kernel_size=1,
                          padding=0,
                          add_bias=True,
                          upsample=False,
                          fused_scale=False,
                          use_wscale=use_wscale,
                          wscale_gain=1.0,
                          activation_type='linear',
                          eps=eps))
            self.pth_to_tf_var_mapping[f'output{block_idx}.weight'] = (
                f'ToRGB_lod{self.final_res_log2 - res_log2}/weight')
            self.pth_to_tf_var_mapping[f'output{block_idx}.bias'] = (
                f'ToRGB_lod{self.final_res_log2 - res_log2}/bias')

    def get_nf(self, res):
        """Gets number of feature maps according to the given resolution."""
        return min(self.fmaps_base // res, self.fmaps_max)

    def forward(self, z, label=None, lod=None):
        if z.ndim != 2 or z.shape[1] != self.z_dim:
            raise ValueError(f'Input latent code should be with shape '
                             f'[batch_size, latent_dim], where '
                             f'`latent_dim` equals to {self.z_dim}!\n'
                             f'But `{z.shape}` is received!')
        z = self.layer0.pixel_norm(z)
        if self.label_dim:
            if label is None:
                raise ValueError(f'Model requires an additional label '
                                 f'(with size {self.label_dim}) as input, '
                                 f'but no label is received!')
            if label.ndim != 2 or label.shape != (z.shape[0], self.label_dim):
                raise ValueError(f'Input label should be with shape '
                                 f'[batch_size, label_dim], where '
                                 f'`batch_size` equals to that of '
                                 f'latent codes ({z.shape[0]}) and '
                                 f'`label_dim` equals to {self.label_dim}!\n'
                                 f'But `{label.shape}` is received!')
            label = label.to(dtype=torch.float32)
            z = torch.cat((z, label), dim=1)

        lod = self.lod.item() if lod is None else lod
        if lod + self.init_res_log2 > self.final_res_log2:
            raise ValueError(f'Maximum level-of-details (lod) is '
                             f'{self.final_res_log2 - self.init_res_log2}, '
                             f'but `{lod}` is received!')

        x = z.view(z.shape[0], self.z_dim + self.label_dim, 1, 1)
        for res_log2 in range(self.init_res_log2, self.final_res_log2 + 1):
            current_lod = self.final_res_log2 - res_log2
            block_idx = res_log2 - self.init_res_log2
            if lod < current_lod + 1:
                x = getattr(self, f'layer{2 * block_idx}')(x)
                x = getattr(self, f'layer{2 * block_idx + 1}')(x)
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

        results = {
            'z': z,
            'label': label,
            'image': image,
        }
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


class UpsamplingLayer(nn.Module):
    """Implements the upsampling layer.

    Basically, this layer can be used to upsample feature maps with nearest
    neighbor interpolation.
    """

    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def extra_repr(self):
        return f'factor={self.scale_factor}'

    def forward(self, x):
        if self.scale_factor <= 1:
            return x
        return F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')


class ConvLayer(nn.Module):
    """Implements the convolutional layer.

    Basically, this layer executes pixel-wise normalization, upsampling (if
    needed), convolution, and activation in sequence.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 add_bias,
                 upsample,
                 fused_scale,
                 use_wscale,
                 wscale_gain,
                 activation_type,
                 eps):
        """Initializes with layer settings.

        Args:
            in_channels: Number of channels of the input tensor.
            out_channels: Number of channels of the output tensor.
            kernel_size: Size of the convolutional kernels.
            padding: Padding used in convolution.
            add_bias: Whether to add bias onto the convolutional result.
            upsample: Whether to upsample the input tensor before convolution.
            fused_scale: Whether to fused `upsample` and `conv2d` together,
                resulting in `conv2d_transpose`.
            use_wscale: Whether to use weight scaling.
            wscale_gain: Gain factor for weight scaling.
            activation_type: Type of activation.
            eps: A small value to avoid divide overflow.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.add_bias = add_bias
        self.upsample = upsample
        self.fused_scale = fused_scale
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.activation_type = activation_type
        self.eps = eps

        self.pixel_norm = PixelNormLayer(dim=1, eps=eps)

        if upsample and not fused_scale:
            self.up = UpsamplingLayer(scale_factor=2)
        else:
            self.up = nn.Identity()

        if upsample and fused_scale:
            self.use_conv2d_transpose = True
            weight_shape = (in_channels, out_channels, kernel_size, kernel_size)
            self.stride = 2
            self.padding = 1
        else:
            self.use_conv2d_transpose = False
            weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
            self.stride = 1

        fan_in = kernel_size * kernel_size * in_channels
        wscale = wscale_gain / np.sqrt(fan_in)
        if use_wscale:
            self.weight = nn.Parameter(torch.randn(*weight_shape))
            self.wscale = wscale
        else:
            self.weight = nn.Parameter(torch.randn(*weight_shape) * wscale)
            self.wscale = 1.0

        if add_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        assert activation_type in ['linear', 'relu', 'lrelu']

    def extra_repr(self):
        return (f'in_ch={self.in_channels}, '
                f'out_ch={self.out_channels}, '
                f'ksize={self.kernel_size}, '
                f'padding={self.padding}, '
                f'wscale_gain={self.wscale_gain:.3f}, '
                f'bias={self.add_bias}, '
                f'upsample={self.scale_factor}, '
                f'fused_scale={self.fused_scale}, '
                f'act={self.activation_type}')

    def forward(self, x):
        x = self.pixel_norm(x)
        x = self.up(x)
        weight = self.weight
        if self.wscale != 1.0:
            weight = weight * self.wscale
        if self.use_conv2d_transpose:
            weight = F.pad(weight, (1, 1, 1, 1, 0, 0, 0, 0), 'constant', 0.0)
            weight = (weight[:, :, 1:, 1:] + weight[:, :, :-1, 1:] +
                      weight[:, :, 1:, :-1] + weight[:, :, :-1, :-1])
            x = F.conv_transpose2d(x,
                                   weight=weight,
                                   bias=self.bias,
                                   stride=self.stride,
                                   padding=self.padding)
        else:
            x = F.conv2d(x,
                         weight=weight,
                         bias=self.bias,
                         stride=self.stride,
                         padding=self.padding)

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
