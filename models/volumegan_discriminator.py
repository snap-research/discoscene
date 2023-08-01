# python3.7
"""Contains the implementation of discriminator described in VolumeGAN.
Paper: https://arxiv.org/pdf/2112.10759.pdf
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from third_party.stylegan2_official_ops import bias_act
from third_party.stylegan2_official_ops import upfirdn2d
from third_party.stylegan2_official_ops import conv2d_gradfix

__all__ = ['VolumeGANDiscriminator']

# Resolutions allowed.
_RESOLUTIONS_ALLOWED = [8, 16, 32, 64, 128, 256, 512, 1024]

# Architectures allowed.
_ARCHITECTURES_ALLOWED = ['resnet', 'skip', 'origin']


class VolumeGANDiscriminator(nn.Module):
    """Defines the discriminator network in VolumeGAN.

    NOTE: The discriminator takes images with `RGB` channel order and pixel
    range [-1, 1] as inputs.

    Settings for the backbone:

    (1) resolution: The resolution of the input image. (default: -1)
    (2) init_res: The initial resolution to start with convolution. (default: 4)
    (3) image_channels: Number of channels of the input image. (default: 3)
    (4) architecture: Type of architecture. Support `origin`, `skip`, and
        `resnet`. (default: `resnet`)
    (5) use_wscale: Whether to use weight scaling. (default: True)
    (6) wscale_gain: The factor to control weight scaling. (default: 1.0)
    (7) lr_mul: Learning rate multiplier for backbone. (default: 1.0)
    (8) mbstd_groups: Group size for the minibatch standard deviation layer.
        `0` means disable. (default: 4)
    (9) mbstd_channels: Number of new channels (appended to the original feature
        map) after the minibatch standard deviation layer. (default: 1)
    (10) fmaps_base: Factor to control number of feature maps for each layer.
         (default: 32 << 10)
    (11) fmaps_max: Maximum number of feature maps in each layer. (default: 512)
    (12) filter_kernel: Kernel used for filtering (e.g., downsampling).
         (default: (1, 3, 3, 1))
    (13) conv_clamp: A threshold to clamp the output of convolution layers to
         avoid overflow under FP16 training. (default: None)
    (14) eps: A small value to avoid divide overflow. (default: 1e-8)

    Settings for conditional model:

    (1) label_dim: Dimension of the additional label for conditional generation.
        In one-hot conditioning case, it is equal to the number of classes. If
        set to 0, conditioning training will be disabled. (default: 0)
    (2) embedding_dim: Dimension of the embedding space, if needed.
        (default: 512)
    (3) embedding_bias: Whether to add bias to embedding learning.
        (default: True)
    (4) embedding_use_wscale: Whether to use weight scaling for embedding
        learning. (default: True)
    (5) embedding_lr_mul: Learning rate multiplier for the embedding learning.
        (default: 1.0)
    (6) normalize_embedding: Whether to normalize the embedding. (default: True)
    (7) mapping_layers: Number of layers of the additional mapping network after
        embedding. (default: 0)
    (8) mapping_fmaps: Number of hidden channels of the additional mapping
        network after embedding. (default: 512)
    (9) mapping_use_wscale: Whether to use weight scaling for the additional
        mapping network. (default: True)
    (10) mapping_lr_mul: Learning rate multiplier for the additional mapping
         network after embedding. (default: 0.1)

    Runtime settings:

    (1) fp16_res: Layers at resolution higher than (or equal to) this field will
        use `float16` precision for computation. This is merely used for
        acceleration. If set as `None`, all layers will use `float32` by
        default. (default: None)
    (2) impl: Implementation mode of some particular ops, e.g., `filtering`,
        `bias_act`, etc. `cuda` means using the official CUDA implementation
        from StyleGAN2, while `ref` means using the native PyTorch ops.
        (default: `cuda`)
    """

    def __init__(self,
                 # Settings for backbone.
                 resolution=-1,
                 init_res=4,
                 image_channels=3,
                 architecture='resnet',
                 use_wscale=True,
                 wscale_gain=1.0,
                 lr_mul=1.0,
                 mbstd_groups=4,
                 mbstd_channels=1,
                 fmaps_base=32 << 10,
                 fmaps_max=512,
                 filter_kernel=(1, 3, 3, 1),
                 conv_clamp=None,
                 eps=1e-8,
                 # Settings for conditional model.
                 label_dim=0,
                 embedding_dim=512,
                 embedding_bias=True,
                 embedding_use_wscale=True,
                 embedding_lr_mul=1.0,
                 normalize_embedding=True,
                 mapping_layers=0,
                 mapping_fmaps=512,
                 mapping_use_wscale=True,
                 mapping_lr_mul=0.1):
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

        self.init_res = init_res
        self.init_res_log2 = int(np.log2(init_res))
        self.resolution = resolution
        self.final_res_log2 = int(np.log2(resolution))
        self.image_channels = image_channels
        self.architecture = architecture
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.lr_mul = lr_mul
        self.mbstd_groups = mbstd_groups
        self.mbstd_channels = mbstd_channels
        self.fmaps_base = fmaps_base
        self.fmaps_max = fmaps_max
        self.filter_kernel = filter_kernel
        self.conv_clamp = conv_clamp
        self.eps = eps

        self.label_dim = label_dim
        self.embedding_dim = embedding_dim
        self.embedding_bias = embedding_bias
        self.embedding_use_wscale = embedding_use_wscale
        self.embedding_lr_mul = embedding_lr_mul
        self.normalize_embedding = normalize_embedding
        self.mapping_layers = mapping_layers
        self.mapping_fmaps = mapping_fmaps
        self.mapping_use_wscale = mapping_use_wscale
        self.mapping_lr_mul = mapping_lr_mul

        self.pth_to_tf_var_mapping = {}
        self.register_buffer('lod', torch.zeros(()))
        # Embedding for conditional discrimination.
        self.use_embedding = label_dim > 0 and embedding_dim > 0
        if self.use_embedding:
            self.embedding = DenseLayer(in_channels=label_dim,
                                        out_channels=embedding_dim,
                                        add_bias=embedding_bias,
                                        init_bias=0.0,
                                        use_wscale=embedding_use_wscale,
                                        wscale_gain=wscale_gain,
                                        lr_mul=embedding_lr_mul,
                                        activation_type='linear')
            self.pth_to_tf_var_mapping['embedding.weight'] = 'LabelEmbed/weight'
            if self.embedding_bias:
                self.pth_to_tf_var_mapping['embedding.bias'] = 'LabelEmbed/bias'

            if self.normalize_embedding:
                self.norm = PixelNormLayer(dim=1, eps=eps)

            for i in range(mapping_layers):
                in_channels = (embedding_dim if i == 0 else mapping_fmaps)
                out_channels = (embedding_dim if i == (mapping_layers - 1) else
                                mapping_fmaps)
                layer_name = f'mapping{i}'
                self.add_module(layer_name,
                                DenseLayer(in_channels=in_channels,
                                           out_channels=out_channels,
                                           add_bias=True,
                                           init_bias=0.0,
                                           use_wscale=mapping_use_wscale,
                                           wscale_gain=wscale_gain,
                                           lr_mul=mapping_lr_mul,
                                           activation_type='lrelu'))
                self.pth_to_tf_var_mapping[f'{layer_name}.weight'] = (
                    f'Mapping{i}/weight')
                self.pth_to_tf_var_mapping[f'{layer_name}.bias'] = (
                    f'Mapping{i}/bias')

        # Convolutional backbone.
        for res_log2 in range(self.final_res_log2, self.init_res_log2 - 1, -1):
            res = 2 ** res_log2
            in_channels = self.get_nf(res)
            out_channels = self.get_nf(res // 2)
            block_idx = self.final_res_log2 - res_log2

            # Input convolution layer for each resolution (if needed).

            layer_name = f'input{block_idx}'
            self.add_module(layer_name,
                            ConvLayer(in_channels=image_channels,
                                      out_channels=in_channels,
                                      kernel_size=1,
                                      add_bias=True,
                                      scale_factor=1,
                                      filter_kernel=None,
                                      use_wscale=use_wscale,
                                      wscale_gain=wscale_gain,
                                      lr_mul=lr_mul,
                                      activation_type='lrelu',
                                      conv_clamp=conv_clamp))
            self.pth_to_tf_var_mapping[f'{layer_name}.weight'] = (
                f'{res}x{res}/FromRGB/weight')
            self.pth_to_tf_var_mapping[f'{layer_name}.bias'] = (
                f'{res}x{res}/FromRGB/bias')

            # Convolution block for each resolution (except the last one).
            if res != self.init_res:
                # First layer (kernel 3x3) without downsampling.
                layer_name = f'layer{2 * block_idx}'
                self.add_module(layer_name,
                                ConvLayer(in_channels=in_channels,
                                          out_channels=in_channels,
                                          kernel_size=3,
                                          add_bias=True,
                                          scale_factor=1,
                                          filter_kernel=None,
                                          use_wscale=use_wscale,
                                          wscale_gain=wscale_gain,
                                          lr_mul=lr_mul,
                                          activation_type='lrelu',
                                          conv_clamp=conv_clamp))
                self.pth_to_tf_var_mapping[f'{layer_name}.weight'] = (
                    f'{res}x{res}/Conv0/weight')
                self.pth_to_tf_var_mapping[f'{layer_name}.bias'] = (
                    f'{res}x{res}/Conv0/bias')

                # Second layer (kernel 3x3) with downsampling
                layer_name = f'layer{2 * block_idx + 1}'
                self.add_module(layer_name,
                                ConvLayer(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=3,
                                          add_bias=True,
                                          scale_factor=2,
                                          filter_kernel=filter_kernel,
                                          use_wscale=use_wscale,
                                          wscale_gain=wscale_gain,
                                          lr_mul=lr_mul,
                                          activation_type='lrelu',
                                          conv_clamp=conv_clamp))
                self.pth_to_tf_var_mapping[f'{layer_name}.weight'] = (
                    f'{res}x{res}/Conv1_down/weight')
                self.pth_to_tf_var_mapping[f'{layer_name}.bias'] = (
                    f'{res}x{res}/Conv1_down/bias')

                # Residual branch (kernel 1x1) with downsampling, without bias,
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

            # Convolution block for last resolution.
            else:
                self.mbstd = MiniBatchSTDLayer(
                    groups=mbstd_groups, new_channels=mbstd_channels, eps=eps)

                # First layer (kernel 3x3) without downsampling.
                layer_name = f'layer{2 * block_idx}'
                self.add_module(
                    layer_name,
                    ConvLayer(in_channels=in_channels + mbstd_channels,
                              out_channels=in_channels,
                              kernel_size=3,
                              add_bias=True,
                              scale_factor=1,
                              filter_kernel=None,
                              use_wscale=use_wscale,
                              wscale_gain=wscale_gain,
                              lr_mul=lr_mul,
                              activation_type='lrelu',
                              conv_clamp=conv_clamp))
                self.pth_to_tf_var_mapping[f'{layer_name}.weight'] = (
                    f'{res}x{res}/Conv/weight')
                self.pth_to_tf_var_mapping[f'{layer_name}.bias'] = (
                    f'{res}x{res}/Conv/bias')

                # Second layer, as a fully-connected layer.
                layer_name = f'layer{2 * block_idx + 1}'
                self.add_module(layer_name,
                                DenseLayer(in_channels=in_channels * res * res,
                                           out_channels=in_channels,
                                           add_bias=True,
                                           init_bias=0.0,
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
                                         out_channels=(embedding_dim
                                                       if self.use_embedding
                                                       else max(label_dim, 1)),
                                         add_bias=True,
                                         init_bias=0.0,
                                         use_wscale=use_wscale,
                                         wscale_gain=wscale_gain,
                                         lr_mul=lr_mul,
                                         activation_type='linear')
                self.pth_to_tf_var_mapping['output.weight'] = 'Output/weight'
                self.pth_to_tf_var_mapping['output.bias'] = 'Output/bias'

        # Used for downsampling input image for `skip` architecture.
        if self.architecture == 'skip':
            self.register_buffer(
                'filter', upfirdn2d.setup_filter(filter_kernel))

    def get_nf(self, res):
        """Gets number of feature maps according to current resolution."""
        return min(self.fmaps_base // res, self.fmaps_max)

    def forward(self, image, lod=None, label=None, fp16_res=None, impl='cuda'):
        # Check shape.
        expected_shape = (self.image_channels, self.resolution, self.resolution)
        if image.ndim != 4 or image.shape[1:] != expected_shape:
            raise ValueError(f'The input tensor should be with shape '
                             f'[batch_size, channel, height, width], where '
                             f'`channel` equals to {self.image_channels}, '
                             f'`height`, `width` equal to {self.resolution}!\n'
                             f'But `{image.shape}` is received!')
        if self.label_dim > 0:
            if label is None:
                raise ValueError(f'Model requires an additional label '
                                 f'(with dimension {self.label_dim}) as input, '
                                 f'but no label is received!')
            batch_size = image.shape[0]
            if label.ndim != 2 or label.shape != (batch_size, self.label_dim):
                raise ValueError(f'Input label should be with shape '
                                 f'[batch_size, label_dim], where '
                                 f'`batch_size` equals to that of '
                                 f'images ({image.shape[0]}) and '
                                 f'`label_dim` equals to {self.label_dim}!\n'
                                 f'But `{label.shape}` is received!')
            label = label.to(dtype=torch.float32)
            if self.use_embedding:
                embed = self.embedding(label, impl=impl)
                if self.normalize_embedding:
                    embed = self.norm(embed)
                for i in range(self.mapping_layers):
                    embed = getattr(self, f'mapping{i}')(embed, impl=impl)

        # Cast to `torch.float16` if needed.
        if fp16_res is not None and self.resolution >= fp16_res:
            image = image.to(torch.float16)
        
        lod = self.lod.item() if lod is None else lod
        x = self.input0(image, impl=impl)

        for res_log2 in range(self.final_res_log2, self.init_res_log2, -1):
            res = 2 ** res_log2
            # Cast to `torch.float16` if needed.
            if fp16_res is not None and res >= fp16_res:
                x = x.to(torch.float16)
            else:
                x = x.to(torch.float32)

            idx = cur_lod = self.final_res_log2 - res_log2  # Block index

            if cur_lod <= lod < cur_lod + 1:
                x = getattr(self, f'input{idx}')(image, impl=impl)
            elif cur_lod - 1 < lod < cur_lod:
                alpha = lod - np.floor(lod)
                y = getattr(self, f'input{idx}')(image, impl=impl)
                x = y * alpha + x * (1 - alpha)
            if lod < cur_lod + 1:
                if self.architecture == 'skip' and idx > 0:
                    image = upfirdn2d.downsample2d(image, self.filter, impl=impl)
                    # Cast to `torch.float16` if needed.
                    if fp16_res is not None and res >= fp16_res:
                        image = image.to(torch.float16)
                    else:
                        image = image.to(torch.float32)
                    y = getattr(self, f'input{idx}')(image, impl=impl)
                    x = x + y
                if self.architecture == 'resnet':
                    residual = getattr(self, f'residual{idx}')(
                        x, runtime_gain=np.sqrt(0.5), impl=impl)
                    x = getattr(self, f'layer{2 * idx}')(x, impl=impl)
                    x = getattr(self, f'layer{2 * idx + 1}')(
                        x, runtime_gain=np.sqrt(0.5), impl=impl)
                    x = x + residual
                else:
                    x = getattr(self, f'layer{2 * idx}')(x, impl=impl)
                    x = getattr(self, f'layer{2 * idx + 1}')(x, impl=impl)

            if lod > cur_lod:
                image = F.avg_pool2d(
                    image, kernel_size=2, stride=2, padding=0)
        # Final output.
        if fp16_res is not None:  # Always use FP32 for the last block.
            x = x.to(torch.float32)
        if self.architecture == 'skip':
            image = upfirdn2d.downsample2d(image, self.filter, impl=impl)
            if fp16_res is not None:  # Always use FP32 for the last block.
                image = image.to(torch.float32)
            y = getattr(self, f'input{idx}')(image, impl=impl)
            x = x + y
        x = self.mbstd(x)
        x = getattr(self, f'layer{2 * idx + 2}')(x, impl=impl)
        x = getattr(self, f'layer{2 * idx + 3}')(x, impl=impl)
        x = self.output(x, impl=impl)

        if self.use_embedding:
            x = (x * embed).sum(dim=1, keepdim=True)
            x = x / np.sqrt(self.embedding_dim)
        elif self.label_dim > 0:
            x = (x * label).sum(dim=1, keepdim=True)

        results = {
            'score': x,
            'label': label
        }
        if self.use_embedding:
            results['embedding'] = embed
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

        dtype = x.dtype

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

        assert x.dtype == dtype
        return x


class ConvLayer(nn.Module):
    """Implements the convolutional layer.

    If downsampling is needed (i.e., `scale_factor = 2`), the feature map will
    be filtered with `filter_kernel` first.
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
            scale_factor: Scale factor for downsampling. `1` means skip
                downsampling.
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
                kernel_size // 2 + (fw - scale_factor + 1) // 2,
                kernel_size // 2 + (fw - scale_factor) // 2,
                kernel_size // 2 + (fh - scale_factor + 1) // 2,
                kernel_size // 2 + (fh - scale_factor) // 2)

    def extra_repr(self):
        return (f'in_ch={self.in_channels}, '
                f'out_ch={self.out_channels}, '
                f'ksize={self.kernel_size}, '
                f'wscale_gain={self.wscale_gain:.3f}, '
                f'bias={self.add_bias}, '
                f'lr_mul={self.lr_mul:.3f}, '
                f'downsample={self.scale_factor}, '
                f'downsample_filter={self.filter_kernel}, '
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

        if self.scale_factor == 1:  # Native convolution without downsampling.
            padding = self.kernel_size // 2
            x = conv2d_gradfix.conv2d(
                x, weight.to(dtype), stride=1, padding=padding, impl=impl)
        else:  # Convolution with downsampling.
            down = self.scale_factor
            f = self.filter
            padding = self.filter_padding
            # When kernel size = 1, use filtering function for downsampling.
            if self.kernel_size == 1:
                x = upfirdn2d.upfirdn2d(
                    x, f, down=down, padding=padding, impl=impl)
                x = conv2d_gradfix.conv2d(
                    x, weight.to(dtype), stride=1, padding=0, impl=impl)
            # When kernel size != 1, use stride convolution for downsampling.
            else:
                x = upfirdn2d.upfirdn2d(
                    x, f, down=1, padding=padding, impl=impl)
                x = conv2d_gradfix.conv2d(
                    x, weight.to(dtype), stride=down, padding=0, impl=impl)

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
        else:
            x = x.matmul(weight.t())
            x = bias_act.bias_act(x, bias, act=self.activation_type, impl=impl)

        assert x.dtype == dtype
        return x
