# python3.7
"""Contains the implementation of discriminator described in PGGAN.

Paper: https://arxiv.org/pdf/1710.10196.pdf

Official TensorFlow implementation:
https://github.com/tkarras/progressive_growing_of_gans
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['PGGANDiscriminator']

# Resolutions allowed.
_RESOLUTIONS_ALLOWED = [8, 16, 32, 64, 128, 256, 512, 1024]

# Default gain factor for weight scaling.
_WSCALE_GAIN = np.sqrt(2.0)

# pylint: disable=missing-function-docstring

class PGGANDiscriminator(nn.Module):
    """Defines the discriminator network in PGGAN.

    NOTE: The discriminator takes images with `RGB` channel order and pixel
    range [-1, 1] as inputs.

    Settings for the network:

    (1) resolution: The resolution of the input image.
    (2) init_res: Smallest resolution of the convolutional backbone.
        (default: 4)
    (3) image_channels: Number of channels of the input image. (default: 3)
    (4) label_dim: Dimension of the additional label for conditional generation.
        In one-hot conditioning case, it is equal to the number of classes. If
        set to 0, conditioning training will be disabled. (default: 0)
    (5) fused_scale: Whether to fused `conv2d` and `downsample` together,
        resulting in `conv2d` with strides. (default: False)
    (6) use_wscale: Whether to use weight scaling. (default: True)
    (7) wscale_gain: The factor to control weight scaling. (default: sqrt(2.0))
    (8) mbstd_groups: Group size for the minibatch standard deviation layer.
        `0` means disable. (default: 16)
    (9) fmaps_base: Factor to control number of feature maps for each layer.
        (default: 16 << 10)
    (10) fmaps_max: Maximum number of feature maps in each layer. (default: 512)
    (11) eps: A small value to avoid divide overflow. (default: 1e-8)
    """

    def __init__(self,
                 resolution,
                 init_res=4,
                 image_channels=3,
                 label_dim=0,
                 fused_scale=False,
                 use_wscale=True,
                 wscale_gain=np.sqrt(2.0),
                 mbstd_groups=16,
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
        self.image_channels = image_channels
        self.label_dim = label_dim
        self.fused_scale = fused_scale
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.mbstd_groups = mbstd_groups
        self.fmaps_base = fmaps_base
        self.fmaps_max = fmaps_max
        self.eps = eps

        # Level-of-details (used for progressive training).
        self.register_buffer('lod', torch.zeros(()))
        self.pth_to_tf_var_mapping = {'lod': 'lod'}

        for res_log2 in range(self.final_res_log2, self.init_res_log2 - 1, -1):
            res = 2 ** res_log2
            in_channels = self.get_nf(res)
            out_channels = self.get_nf(res // 2)
            block_idx = self.final_res_log2 - res_log2

            # Input convolution layer for each resolution.
            self.add_module(
                f'input{block_idx}',
                ConvLayer(in_channels=self.image_channels,
                          out_channels=in_channels,
                          kernel_size=1,
                          add_bias=True,
                          downsample=False,
                          fused_scale=False,
                          use_wscale=use_wscale,
                          wscale_gain=wscale_gain,
                          activation_type='lrelu'))
            self.pth_to_tf_var_mapping[f'input{block_idx}.weight'] = (
                f'FromRGB_lod{block_idx}/weight')
            self.pth_to_tf_var_mapping[f'input{block_idx}.bias'] = (
                f'FromRGB_lod{block_idx}/bias')

            # Convolution block for each resolution (except the last one).
            if res != self.init_res:
                self.add_module(
                    f'layer{2 * block_idx}',
                    ConvLayer(in_channels=in_channels,
                              out_channels=in_channels,
                              kernel_size=3,
                              add_bias=True,
                              downsample=False,
                              fused_scale=False,
                              use_wscale=use_wscale,
                              wscale_gain=wscale_gain,
                              activation_type='lrelu'))
                tf_layer0_name = 'Conv0'
                self.add_module(
                    f'layer{2 * block_idx + 1}',
                    ConvLayer(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              add_bias=True,
                              downsample=True,
                              fused_scale=fused_scale,
                              use_wscale=use_wscale,
                              wscale_gain=wscale_gain,
                              activation_type='lrelu'))
                tf_layer1_name = 'Conv1_down' if fused_scale else 'Conv1'

            # Convolution block for last resolution.
            else:
                self.mbstd = MiniBatchSTDLayer(groups=mbstd_groups, eps=eps)
                self.add_module(
                    f'layer{2 * block_idx}',
                    ConvLayer(
                        in_channels=in_channels + 1,
                        out_channels=in_channels,
                        kernel_size=3,
                        add_bias=True,
                        downsample=False,
                        fused_scale=False,
                        use_wscale=use_wscale,
                        wscale_gain=wscale_gain,
                        activation_type='lrelu'))
                tf_layer0_name = 'Conv'
                self.add_module(
                    f'layer{2 * block_idx + 1}',
                    DenseLayer(in_channels=in_channels * res * res,
                               out_channels=out_channels,
                               add_bias=True,
                               use_wscale=use_wscale,
                               wscale_gain=wscale_gain,
                               activation_type='lrelu'))
                tf_layer1_name = 'Dense0'

            self.pth_to_tf_var_mapping[f'layer{2 * block_idx}.weight'] = (
                f'{res}x{res}/{tf_layer0_name}/weight')
            self.pth_to_tf_var_mapping[f'layer{2 * block_idx}.bias'] = (
                f'{res}x{res}/{tf_layer0_name}/bias')
            self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 1}.weight'] = (
                f'{res}x{res}/{tf_layer1_name}/weight')
            self.pth_to_tf_var_mapping[f'layer{2 * block_idx + 1}.bias'] = (
                f'{res}x{res}/{tf_layer1_name}/bias')

        # Final dense layer.
        self.output = DenseLayer(in_channels=out_channels,
                                 out_channels=1 + self.label_dim,
                                 add_bias=True,
                                 use_wscale=self.use_wscale,
                                 wscale_gain=1.0,
                                 activation_type='linear')
        self.pth_to_tf_var_mapping['output.weight'] = (
            f'{res}x{res}/Dense1/weight')
        self.pth_to_tf_var_mapping['output.bias'] = (
            f'{res}x{res}/Dense1/bias')

    def get_nf(self, res):
        """Gets number of feature maps according to the given resolution."""
        return min(self.fmaps_base // res, self.fmaps_max)

    def forward(self, image, lod=None):
        expected_shape = (self.image_channels, self.resolution, self.resolution)
        if image.ndim != 4 or image.shape[1:] != expected_shape:
            raise ValueError(f'The input tensor should be with shape '
                             f'[batch_size, channel, height, width], where '
                             f'`channel` equals to {self.image_channels}, '
                             f'`height`, `width` equal to {self.resolution}!\n'
                             f'But `{image.shape}` is received!')

        lod = self.lod.item() if lod is None else lod
        if lod + self.init_res_log2 > self.final_res_log2:
            raise ValueError(f'Maximum level-of-details (lod) is '
                             f'{self.final_res_log2 - self.init_res_log2}, '
                             f'but `{lod}` is received!')

        lod = self.lod.item()
        for res_log2 in range(self.final_res_log2, self.init_res_log2 - 1, -1):
            block_idx = current_lod = self.final_res_log2 - res_log2
            if current_lod <= lod < current_lod + 1:
                x = getattr(self, f'input{block_idx}')(image)
            elif current_lod - 1 < lod < current_lod:
                alpha = lod - np.floor(lod)
                y = getattr(self, f'input{block_idx}')(image)
                x = y * alpha + x * (1 - alpha)
            if lod < current_lod + 1:
                if res_log2 == self.init_res_log2:
                    x = self.mbstd(x)
                x = getattr(self, f'layer{2 * block_idx}')(x)
                x = getattr(self, f'layer{2 * block_idx + 1}')(x)
            if lod > current_lod:
                image = F.avg_pool2d(
                    image, kernel_size=2, stride=2, padding=0)
        x = self.output(x)

        return {'score': x}


class MiniBatchSTDLayer(nn.Module):
    """Implements the minibatch standard deviation layer."""

    def __init__(self, groups, eps):
        super().__init__()
        self.groups = groups
        self.eps = eps

    def extra_repr(self):
        return f'groups={self.groups}, epsilon={self.eps}'

    def forward(self, x):
        if self.groups <= 1:
            return x

        N, C, H, W = x.shape
        G = min(self.groups, N)  # Number of groups.

        y = x.reshape(G, -1, C, H, W)            # [GnCHW]
        y = y - y.mean(dim=0)                    # [GnCHW]
        y = y.square().mean(dim=0)               # [nCHW]
        y = (y + self.eps).sqrt()                # [nCHW]
        y = y.mean(dim=(1, 2, 3), keepdim=True)  # [n111]
        y = y.repeat(G, 1, H, W)                 # [N1HW]
        x = torch.cat([x, y], dim=1)             # [N(C+1)HW]

        return x


class DownsamplingLayer(nn.Module):
    """Implements the downsampling layer.

    Basically, this layer can be used to downsample feature maps with average
    pooling.
    """

    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def extra_repr(self):
        return f'factor={self.scale_factor}'

    def forward(self, x):
        if self.scale_factor <= 1:
            return x
        return F.avg_pool2d(x,
                            kernel_size=self.scale_factor,
                            stride=self.scale_factor,
                            padding=0)


class ConvLayer(nn.Module):
    """Implements the convolutional layer.

    Basically, this layer executes convolution, activation, and downsampling (if
    needed) in sequence.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 add_bias,
                 downsample,
                 fused_scale,
                 use_wscale,
                 wscale_gain,
                 activation_type):
        """Initializes with layer settings.

        Args:
            in_channels: Number of channels of the input tensor.
            out_channels: Number of channels of the output tensor.
            kernel_size: Size of the convolutional kernels.
            add_bias: Whether to add bias onto the convolutional result.
            downsample: Whether to downsample the result after convolution.
            fused_scale: Whether to fused `conv2d` and `downsample` together,
                resulting in `conv2d` with strides.
            use_wscale: Whether to use weight scaling.
            wscale_gain: Gain factor for weight scaling.
            activation_type: Type of activation.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.add_bias = add_bias
        self.downsample = downsample
        self.fused_scale = fused_scale
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.activation_type = activation_type

        if downsample and not fused_scale:
            self.down = DownsamplingLayer(scale_factor=2)
        else:
            self.down = nn.Identity()

        if downsample and fused_scale:
            self.use_stride = True
            self.stride = 2
            self.padding = 1
        else:
            self.use_stride = False
            self.stride = 1
            self.padding = kernel_size // 2

        weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
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
                f'wscale_gain={self.wscale_gain:.3f}, '
                f'bias={self.add_bias}, '
                f'downsample={self.scale_factor}, '
                f'fused_scale={self.fused_scale}, '
                f'act={self.activation_type}')

    def forward(self, x):
        weight = self.weight
        if self.wscale != 1.0:
            weight = weight * self.wscale

        if self.use_stride:
            weight = F.pad(weight, (1, 1, 1, 1, 0, 0, 0, 0), 'constant', 0.0)
            weight = (weight[:, :, 1:, 1:] + weight[:, :, :-1, 1:] +
                      weight[:, :, 1:, :-1] + weight[:, :, :-1, :-1]) * 0.25
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
        x = self.down(x)

        return x


class DenseLayer(nn.Module):
    """Implements the dense layer."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 add_bias,
                 use_wscale,
                 wscale_gain,
                 activation_type):
        """Initializes with layer settings.

        Args:
            in_channels: Number of channels of the input tensor.
            out_channels: Number of channels of the output tensor.
            add_bias: Whether to add bias onto the fully-connected result.
            use_wscale: Whether to use weight scaling.
            wscale_gain: Gain factor for weight scaling.
            activation_type: Type of activation.

        Raises:
            NotImplementedError: If the `activation_type` is not supported.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_bias = add_bias
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.activation_type = activation_type

        weight_shape = (out_channels, in_channels)
        wscale = wscale_gain / np.sqrt(in_channels)
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

    def forward(self, x):
        if x.ndim != 2:
            x = x.flatten(start_dim=1)

        weight = self.weight
        if self.wscale != 1.0:
            weight = weight * self.wscale

        x = F.linear(x, weight=weight, bias=self.bias)

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
