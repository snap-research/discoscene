# python3.7
"""Contains the implementation of discriminator described in StyleGAN.

Paper: https://arxiv.org/pdf/1812.04948.pdf

Official TensorFlow implementation: https://github.com/NVlabs/stylegan
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

__all__ = ['StyleGANDiscriminator']

# Resolutions allowed.
_RESOLUTIONS_ALLOWED = [8, 16, 32, 64, 128, 256, 512, 1024]

# Fused-scale options allowed.
_FUSED_SCALE_ALLOWED = [True, False, 'auto']

# pylint: disable=missing-function-docstring

class StyleGANDiscriminator(nn.Module):
    """Defines the discriminator network in StyleGAN.

    NOTE: The discriminator takes images with `RGB` channel order and pixel
    range [-1, 1] as inputs.

    Settings for the backbone:

    (1) resolution: The resolution of the input image. (default: -1)
    (2) init_res: Smallest resolution of the convolutional backbone.
        (default: 4)
    (3) image_channels: Number of channels of the input image. (default: 3)
    (4) fused_scale:  The strategy of fusing `conv2d` and `downsample` as one
        operator. `True` means blocks from all resolutions will fuse. `False`
        means blocks from all resolutions will not fuse. `auto` means blocks
        from resolutions higher than (or equal to) `fused_scale_res` will fuse.
        (default: `auto`)
    (5) fused_scale_res: Minimum resolution to fuse `conv2d` and `downsample`
        as one operator. This field only takes effect if `fused_scale` is set
        as `auto`. (default: 128)
    (6) use_wscale: Whether to use weight scaling. (default: True)
    (7) wscale_gain: The factor to control weight scaling. (default: sqrt(2.0))
    (8) lr_mul: Learning rate multiplier for backbone. (default: 1.0)
    (9) mbstd_groups: Group size for the minibatch standard deviation layer.
        `0` means disable. (default: 4)
    (10) mbstd_channels: Number of new channels (appended to the original
         feature map) after the minibatch standard deviation layer. (default: 1)
    (11) fmaps_base: Factor to control number of feature maps for each layer.
         (default: 16 << 10)
    (12) fmaps_max: Maximum number of feature maps in each layer. (default: 512)
    (13) filter_kernel: Kernel used for filtering (e.g., downsampling).
         (default: (1, 2, 1))
    (14) eps: A small value to avoid divide overflow. (default: 1e-8)

    Settings for conditional model:

    (1) label_dim: Dimension of the additional label for conditional generation.
        In one-hot conditioning case, it is equal to the number of classes. If
        set to 0, conditioning training will be disabled. (default: 0)

    Runtime settings:

    (1) enable_amp: Whether to enable automatic mixed precision training.
        (default: False)
    """

    def __init__(self,
                 # Settings for backbone.
                 resolution=-1,
                 init_res=4,
                 image_channels=3,
                 fused_scale='auto',
                 fused_scale_res=128,
                 use_wscale=True,
                 wscale_gain=np.sqrt(2.0),
                 lr_mul=1.0,
                 mbstd_groups=4,
                 mbstd_channels=1,
                 fmaps_base=16 << 10,
                 fmaps_max=512,
                 filter_kernel=(1, 2, 1),
                 eps=1e-8,
                 # Settings for conditional model.
                 label_dim=0):
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

        self.init_res = init_res
        self.init_res_log2 = int(np.log2(init_res))
        self.resolution = resolution
        self.final_res_log2 = int(np.log2(resolution))
        self.image_channels = image_channels
        self.fused_scale = fused_scale
        self.fused_scale_res = fused_scale_res
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.lr_mul = lr_mul
        self.mbstd_groups = mbstd_groups
        self.mbstd_channels = mbstd_channels
        self.fmaps_base = fmaps_base
        self.fmaps_max = fmaps_max
        self.filter_kernel = filter_kernel
        self.eps = eps
        self.label_dim = label_dim

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
                ConvLayer(in_channels=image_channels,
                          out_channels=in_channels,
                          kernel_size=1,
                          add_bias=True,
                          scale_factor=1,
                          fused_scale=False,
                          filter_kernel=None,
                          use_wscale=use_wscale,
                          wscale_gain=wscale_gain,
                          lr_mul=lr_mul,
                          activation_type='lrelu'))
            self.pth_to_tf_var_mapping[f'input{block_idx}.weight'] = (
                f'FromRGB_lod{block_idx}/weight')
            self.pth_to_tf_var_mapping[f'input{block_idx}.bias'] = (
                f'FromRGB_lod{block_idx}/bias')

            # Convolution block for each resolution (except the last one).
            if res != self.init_res:
                # First layer (kernel 3x3) without downsampling.
                layer_name = f'layer{2 * block_idx}'
                self.add_module(
                    layer_name,
                    ConvLayer(in_channels=in_channels,
                              out_channels=in_channels,
                              kernel_size=3,
                              add_bias=True,
                              scale_factor=1,
                              fused_scale=False,
                              filter_kernel=None,
                              use_wscale=use_wscale,
                              wscale_gain=wscale_gain,
                              lr_mul=lr_mul,
                              activation_type='lrelu'))
                self.pth_to_tf_var_mapping[f'{layer_name}.weight'] = (
                    f'{res}x{res}/Conv0/weight')
                self.pth_to_tf_var_mapping[f'{layer_name}.bias'] = (
                    f'{res}x{res}/Conv0/bias')

                # Second layer (kernel 3x3) with downsampling
                layer_name = f'layer{2 * block_idx + 1}'
                self.add_module(
                    layer_name,
                    ConvLayer(in_channels=in_channels,
                              out_channels=out_channels,
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
                              activation_type='lrelu'))
                self.pth_to_tf_var_mapping[f'{layer_name}.weight'] = (
                    f'{res}x{res}/Conv1_down/weight')
                self.pth_to_tf_var_mapping[f'{layer_name}.bias'] = (
                    f'{res}x{res}/Conv1_down/bias')

            # Convolution block for last resolution.
            else:
                self.mbstd = MiniBatchSTDLayer(groups=mbstd_groups,
                                               new_channels=mbstd_channels,
                                               eps=eps)

                # First layer (kernel 3x3) without downsampling.
                layer_name = f'layer{2 * block_idx}'
                self.add_module(
                    layer_name,
                    ConvLayer(in_channels=in_channels + mbstd_channels,
                              out_channels=in_channels,
                              kernel_size=3,
                              add_bias=True,
                              scale_factor=1,
                              fused_scale=False,
                              filter_kernel=None,
                              use_wscale=use_wscale,
                              wscale_gain=wscale_gain,
                              lr_mul=lr_mul,
                              activation_type='lrelu'))
                self.pth_to_tf_var_mapping[f'{layer_name}.weight'] = (
                    f'{res}x{res}/Conv/weight')
                self.pth_to_tf_var_mapping[f'{layer_name}.bias'] = (
                    f'{res}x{res}/Conv/bias')

                # Second layer, as a fully-connected layer.
                layer_name = f'layer{2 * block_idx + 1}'
                self.add_module(
                    f'layer{2 * block_idx + 1}',
                    DenseLayer(in_channels=in_channels * res * res,
                               out_channels=in_channels,
                               add_bias=True,
                               use_wscale=use_wscale,
                               wscale_gain=wscale_gain,
                               lr_mul=lr_mul,
                               activation_type='lrelu'))
                self.pth_to_tf_var_mapping[f'{layer_name}.weight'] = (
                    f'{res}x{res}/Dense0/weight')
                self.pth_to_tf_var_mapping[f'{layer_name}.bias'] = (
                    f'{res}x{res}/Dense0/bias')

                # Final dense layer to output score.
                self.output = DenseLayer(in_channels=in_channels,
                                         out_channels=max(label_dim, 1),
                                         add_bias=True,
                                         use_wscale=use_wscale,
                                         wscale_gain=1.0,
                                         lr_mul=lr_mul,
                                         activation_type='linear')
                self.pth_to_tf_var_mapping['output.weight'] = (
                    f'{res}x{res}/Dense1/weight')
                self.pth_to_tf_var_mapping['output.bias'] = (
                    f'{res}x{res}/Dense1/bias')

    def get_nf(self, res):
        """Gets number of feature maps according to the given resolution."""
        return min(self.fmaps_base // res, self.fmaps_max)

    def forward(self, image, label=None, lod=None, enable_amp=False):
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

        if self.label_dim:
            if label is None:
                raise ValueError(f'Model requires an additional label '
                                 f'(with dimension {self.label_dim}) as input, '
                                 f'but no label is received!')
            batch = image.shape[0]
            if (label.ndim != 2 or label.shape != (batch, self.label_dim)):
                raise ValueError(f'Input label should be with shape '
                                 f'[batch_size, label_dim], where '
                                 f'`batch_size` equals to {batch}, and '
                                 f'`label_dim` equals to {self.label_dim}!\n'
                                 f'But `{label.shape}` is received!')
            label = label.to(dtype=torch.float32)

        with autocast(enabled=enable_amp):
            for res_log2 in range(
                    self.final_res_log2, self.init_res_log2 - 1, -1):
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

            if self.label_dim:
                x = (x * label).sum(dim=1, keepdim=True)

        results = {
            'score': x,
            'label': label
        }
        return results


class MiniBatchSTDLayer(nn.Module):
    """Implements the minibatch standard deviation layer."""

    def __init__(self, groups, new_channels, eps):
        super().__init__()
        self.groups = groups
        self.new_channels = new_channels
        self.eps = eps

    def extra_repr(self):
        return (f'groups={self.groups}, '
                f'new_channels={self.new_channels}, '
                f'epsilon={self.eps}')

    def forward(self, x):
        if self.groups <= 1 or self.new_channels < 1:
            return x

        N, C, H, W = x.shape
        G = min(self.groups, N)  # Number of groups.
        nC = self.new_channels  # Number of channel groups.
        c = C // nC             # Channels per channel group.

        y = x.reshape(G, -1, nC, c, H, W)  # [GnFcHW]
        y = y - y.mean(dim=0)              # [GnFcHW]
        y = y.square().mean(dim=0)         # [nFcHW]
        y = (y + self.eps).sqrt()          # [nFcHW]
        y = y.mean(dim=(2, 3, 4))          # [nF]
        y = y.reshape(-1, nC, 1, 1)        # [nF11]
        y = y.repeat(G, 1, H, W)           # [NFHW]
        x = torch.cat((x, y), dim=1)       # [N(C+F)HW]

        return x


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
        dx = BlurBackPropagation.apply(dy, kernel)
        return dx, None, None


class BlurBackPropagation(torch.autograd.Function):
    """Defines the back propagation of blur operation.

    NOTE: This is used to speed up the backward of gradient penalty.
    """

    @staticmethod
    def forward(ctx, dy, kernel):
        ctx.save_for_backward(kernel)
        dx = F.conv2d(input=dy,
                      weight=kernel.flip((2, 3)),
                      bias=None,
                      stride=1,
                      padding=1,
                      groups=dy.shape[1])
        return dx

    @staticmethod
    def backward(ctx, ddx):
        kernel, = ctx.saved_tensors
        ddy = F.conv2d(input=ddx,
                       weight=kernel,
                       bias=None,
                       stride=1,
                       padding=1,
                       groups=ddx.shape[1])
        return ddy, None, None


class ConvLayer(nn.Module):
    """Implements the convolutional layer.

    If downsampling is needed (i.e., `scale_factor = 2`), the feature map will
    be filtered with `filter_kernel` first. If `fused_scale` is set as `True`,
    `conv2d` and `downsample` will be fused as one operator, using stride
    convolution.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 add_bias,
                 scale_factor,
                 fused_scale,
                 filter_kernel,
                 use_wscale,
                 wscale_gain,
                 lr_mul,
                 activation_type):
        """Initializes with layer settings.

        Args:
            in_channels: Number of channels of the input tensor.
            out_channels: Number of channels of the output tensor.
            kernel_size: Size of the convolutional kernels.
            add_bias: Whether to add bias onto the convolutional result.
            scale_factor: Scale factor for downsampling. `1` means skip
                downsampling.
            fused_scale: Whether to fuse `conv2d` and `downsample` as one
                operator, using stride convolution.
            filter_kernel: Kernel used for filtering.
            use_wscale: Whether to use weight scaling.
            wscale_gain: Gain factor for weight scaling.
            lr_mul: Learning multiplier for both weight and bias.
            activation_type: Type of activation.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.add_bias = add_bias
        self.scale_factor = scale_factor
        self.fused_scale = fused_scale
        self.filter_kernel = filter_kernel
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.lr_mul = lr_mul
        self.activation_type = activation_type

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

        if scale_factor > 1:
            assert filter_kernel is not None
            kernel = np.array(filter_kernel, dtype=np.float32).reshape(1, -1)
            kernel = kernel.T.dot(kernel)
            kernel = kernel / np.sum(kernel)
            kernel = kernel[np.newaxis, np.newaxis]
            self.register_buffer('filter', torch.from_numpy(kernel))

        if scale_factor > 1 and fused_scale:  # use stride convolution.
            self.stride = scale_factor
        else:
            self.stride = 1
        self.padding = kernel_size // 2

        assert activation_type in ['linear', 'relu', 'lrelu']

    def extra_repr(self):
        return (f'in_ch={self.in_channels}, '
                f'out_ch={self.out_channels}, '
                f'ksize={self.kernel_size}, '
                f'wscale_gain={self.wscale_gain:.3f}, '
                f'bias={self.add_bias}, '
                f'lr_mul={self.lr_mul:.3f}, '
                f'downsample={self.scale_factor}, '
                f'fused_scale={self.fused_scale}, '
                f'downsample_filter={self.filter_kernel}, '
                f'act={self.activation_type}')

    def forward(self, x):
        if self.scale_factor > 1:
            # Disable `autocast` for customized autograd function.
            # Please check reference:
            # https://pytorch.org/docs/stable/notes/amp_examples.html#autocast-and-custom-autograd-functions
            with autocast(enabled=False):
                f = self.filter.repeat(self.in_channels, 1, 1, 1)
                x = Blur.apply(x.float(), f)  # Always use FP32.

        weight = self.weight
        if self.wscale != 1.0:
            weight = weight * self.wscale
        bias = None
        if self.bias is not None:
            bias = self.bias
            if self.bscale != 1.0:
                bias = bias * self.bscale

        if self.scale_factor > 1 and self.fused_scale:
            weight = F.pad(weight, (1, 1, 1, 1, 0, 0, 0, 0), 'constant', 0.0)
            weight = (weight[:, :, 1:, 1:] + weight[:, :, :-1, 1:] +
                      weight[:, :, 1:, :-1] + weight[:, :, :-1, :-1]) * 0.25
        x = F.conv2d(x,
                     weight=weight,
                     bias=bias,
                     stride=self.stride,
                     padding=self.padding)
        if self.scale_factor > 1 and not self.fused_scale:
            down = self.scale_factor
            x = F.avg_pool2d(x, kernel_size=down, stride=down, padding=0)

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
