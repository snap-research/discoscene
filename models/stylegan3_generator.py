# python3.7
"""Contains the implementation of generator described in StyleGAN3.

Compared to that of StyleGAN2, the generator in StyleGAN3 controls the frequency
flow along with the convolutional layers growing.

Paper: https://arxiv.org/pdf/2106.12423.pdf

Official implementation: https://github.com/NVlabs/stylegan3
"""

import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F

from third_party.stylegan3_official_ops import bias_act
from third_party.stylegan3_official_ops import filtered_lrelu
from third_party.stylegan3_official_ops import conv2d_gradfix
from .utils.ops import all_gather

__all__ = ['StyleGAN3Generator']

# Resolutions allowed.
_RESOLUTIONS_ALLOWED = [8, 16, 32, 64, 128, 256, 512, 1024]

# pylint: disable=missing-function-docstring

class StyleGAN3Generator(nn.Module):
    """Defines the generator network in StyleGAN3.

    NOTE: The synthesized images are with `RGB` channel order and pixel range
    [-1, 1].

    Settings for the mapping network:

    (1) z_dim: Dimension of the input latent space, Z. (default: 512)
    (2) w_dim: Dimension of the output latent space, W. (default: 512)
    (3) repeat_w: Repeat w-code for different layers. (default: True)
    (4) normalize_z: Whether to normalize the z-code. (default: True)
    (5) mapping_layers: Number of layers of the mapping network. (default: 2)
    (6) mapping_fmaps: Number of hidden channels of the mapping network.
        (default: 512)
    (7) mapping_lr_mul: Learning rate multiplier for the mapping network.
        (default: 0.01)

    Settings for conditional generation:

    (1) label_dim: Dimension of the additional label for conditional generation.
        In one-hot conditioning case, it is equal to the number of classes. If
        set to 0, conditioning training will be disabled. (default: 0)
    (2) embedding_dim: Dimension of the embedding space, if needed.
        (default: 512)
    (3) embedding_bias: Whether to add bias to embedding learning.
        (default: True)
    (4) embedding_lr_mul: Learning rate multiplier for the embedding learning.
        (default: 1.0)
    (5) normalize_embedding: Whether to normalize the embedding. (default: True)
    (6) normalize_embedding_latent: Whether to normalize the embedding together
        with the latent. (default: False)

    Settings for the synthesis network:

    (1) resolution: The resolution of the output image. (default: -1)
    (2) image_channels: Number of channels of the output image. (default: 3)
    (3) final_tanh: Whether to use `tanh` to control the final pixel range.
        (default: False)
    (4) output_scale: Factor to scaling the output image. (default: 0.25)
    (5) num_layers: Number of synthesis layers, excluding the first positional
        encoding layer and the last ToRGB layer. (default: 14)
    (6) num_critical: Number of synthesis layers with critical sampling. These
        layers are always set as top (with highest resolution) ones.
    (7) fmaps_base: Factor to control number of feature maps for each layer.
         (default: 32 << 10)
    (8) fmaps_max: Maximum number of feature maps in each layer. (default: 512)
    (9) kernel_size: Size of convolutional kernels. (default: 1)
    (10) conv_clamp: A threshold to clamp the output of convolution layers to
         avoid overflow under FP16 training. (default: None)
    (11) first_cutoff: Cutoff frequency of the first layer. (default: 2)
    (12) first_stopband: Stopband of the first layer. (default: 2 ** 2.1)
    (13) last_stopband_rel: Stopband of the last layer, relative to the last
         cutoff, which is `resolution / 2`. Concretely, `last_stopband` will be
         equal to `resolution / 2 * last_stopband_rel`. (default: 2 ** 0.3)
    (14) margin_size: Size of margin for each feature map. (default: 10)
    (15) filter_size: Size of filter for upsampling and downsampling around the
         activation. (default: 6)
    (16) act_upsampling: Factor used to upsample the feature map before
         activation for anti-aliasing. (default: 2)
    (17) use_radial_filter: Whether to use radial filter for downsampling after
         the activation. (default: False)
    (18) eps: A small value to avoid divide overflow. (default: 1e-8)

    Runtime settings:

    (1) w_moving_decay: Decay factor for updating `w_avg`, which is used for
        training only. Set `None` to disable. (default: 0.998)
    (2) sync_w_avg: Synchronizing the stats of `w_avg` across replicas. If set
        as `True`, the stats will be more accurate, yet the speed maybe a little
        bit slower. (default: False)
    (3) style_mixing_prob: Probability to perform style mixing as a training
        regularization. Set `None` to disable. (default: None)
    (4) trunc_psi: Truncation psi, set `None` to disable. (default: None)
    (5) trunc_layers: Number of layers to perform truncation. (default: None)
    (6) magnitude_moving_decay: Decay factor for updating `magnitude_ema` in
        each `SynthesisLayer`, which is used for training only. Set `None` to
        disable. (default: 0.999)
    (7) update_ema: Whether to update `w_avg` in the `MappingNetwork` and
        `magnitude_ema` in each `SynthesisLayer`. This field only takes effect
        in `training` model. (default: False)
    (8) fp16_res: Layers at resolution higher than (or equal to) this field will
        use `float16` precision for computation. This is merely used for
        acceleration. If set as `None`, all layers will use `float32` by
        default. (default: None)
    (9) impl: Implementation mode of some particular ops, e.g., `filtering`,
        `bias_act`, etc. `cuda` means using the official CUDA implementation
        from StyleGAN3, while `ref` means using the native PyTorch ops.
        (default: `cuda`)
    """

    def __init__(self,
                 # Settings for mapping network.
                 z_dim=512,
                 w_dim=512,
                 repeat_w=True,
                 normalize_z=True,
                 mapping_layers=2,
                 mapping_fmaps=512,
                 mapping_lr_mul=0.01,
                 # Settings for conditional generation.
                 label_dim=0,
                 embedding_dim=512,
                 embedding_bias=True,
                 embedding_lr_mul=1.0,
                 normalize_embedding=True,
                 normalize_embedding_latent=False,
                 # Settings for synthesis network.
                 resolution=-1,
                 image_channels=3,
                 final_tanh=False,
                 output_scale=0.25,
                 num_layers=14,
                 num_critical=2,
                 fmaps_base=32 << 10,
                 fmaps_max=512,
                 kernel_size=1,
                 conv_clamp=256,
                 first_cutoff=2,
                 first_stopband=2 ** 2.1,
                 last_stopband_rel=2 ** 0.3,
                 margin_size=10,
                 filter_size=6,
                 act_upsampling=2,
                 use_radial_filter=False,
                 eps=1e-8):
        """Initializes with basic settings."""
        super().__init__()

        if resolution not in _RESOLUTIONS_ALLOWED:
            raise ValueError(f'Invalid resolution: `{resolution}`!\n'
                             f'Resolutions allowed: {_RESOLUTIONS_ALLOWED}.')

        self.z_dim = z_dim
        self.w_dim = w_dim
        self.repeat_w = repeat_w
        self.normalize_z = normalize_z
        self.mapping_layers = mapping_layers
        self.mapping_fmaps = mapping_fmaps
        self.mapping_lr_mul = mapping_lr_mul

        self.label_dim = label_dim
        self.embedding_dim = embedding_dim
        self.embedding_bias = embedding_bias
        self.embedding_lr_mul = embedding_lr_mul
        self.normalize_embedding = normalize_embedding
        self.normalize_embedding_latent = normalize_embedding_latent

        self.resolution = resolution
        self.image_channels = image_channels
        self.final_tanh = final_tanh
        self.output_scale = output_scale
        self.num_layers = num_layers + 2  # Including InputLayer and ToRGBLayer.
        self.num_critical = num_critical
        self.fmaps_base = fmaps_base
        self.fmaps_max = fmaps_max
        self.kernel_size = kernel_size
        self.conv_clamp = conv_clamp
        self.first_cutoff = first_cutoff
        self.first_stopband = first_stopband
        self.last_stopband_rel = last_stopband_rel
        self.margin_size = margin_size
        self.filter_size = filter_size
        self.act_upsampling = act_upsampling
        self.use_radial_filter = use_radial_filter
        self.eps = eps

        # Dimension of latent space, which is convenient for sampling.
        self.latent_dim = (z_dim,)

        self.mapping = MappingNetwork(
            input_dim=z_dim,
            output_dim=w_dim,
            num_outputs=self.num_layers,
            repeat_output=repeat_w,
            normalize_input=normalize_z,
            num_layers=mapping_layers,
            hidden_dim=mapping_fmaps,
            lr_mul=mapping_lr_mul,
            label_dim=label_dim,
            embedding_dim=embedding_dim,
            embedding_bias=embedding_bias,
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
                                          w_dim=w_dim,
                                          image_channels=image_channels,
                                          final_tanh=final_tanh,
                                          output_scale=output_scale,
                                          num_layers=num_layers,
                                          num_critical=num_critical,
                                          fmaps_base=fmaps_base,
                                          fmaps_max=fmaps_max,
                                          kernel_size=kernel_size,
                                          conv_clamp=conv_clamp,
                                          first_cutoff=first_cutoff,
                                          first_stopband=first_stopband,
                                          last_stopband_rel=last_stopband_rel,
                                          margin_size=margin_size,
                                          filter_size=filter_size,
                                          act_upsampling=act_upsampling,
                                          use_radial_filter=use_radial_filter,
                                          eps=eps)

        self.var_mapping = {'w_avg': 'mapping.w_avg'}
        for key, val in self.mapping.var_mapping.items():
            self.var_mapping[f'mapping.{key}'] = f'mapping.{val}'
        for key, val in self.synthesis.var_mapping.items():
            self.var_mapping[f'synthesis.{key}'] = f'synthesis.{val}'

    def set_space_of_latent(self, space_of_latent):
        """Sets the space to which the latent code belong.

        See `SynthesisNetwork` for more details.
        """
        self.synthesis.set_space_of_latent(space_of_latent)

    def forward(self,
                z,
                label=None,
                w_moving_decay=0.998,
                sync_w_avg=False,
                style_mixing_prob=None,
                trunc_psi=None,
                trunc_layers=None,
                magnitude_moving_decay=0.999,
                update_ema=False,
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
        if self.training and update_ema and w_moving_decay is not None:
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

        synthesis_results = self.synthesis(
            wp,
            magnitude_moving_decay=magnitude_moving_decay,
            update_ema=update_ema,
            fp16_res=fp16_res,
            impl=impl)

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
                 lr_mul,
                 label_dim,
                 embedding_dim,
                 embedding_bias,
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
        self.lr_mul = lr_mul
        self.label_dim = label_dim
        self.embedding_dim = embedding_dim
        self.embedding_bias = embedding_bias
        self.embedding_lr_mul = embedding_lr_mul
        self.normalize_embedding = normalize_embedding
        self.normalize_embedding_latent = normalize_embedding_latent
        self.eps = eps

        self.var_mapping = {}

        self.norm = PixelNormLayer(dim=1, eps=eps)

        if self.label_dim > 0:
            input_dim = input_dim + embedding_dim
            self.embedding = DenseLayer(in_channels=label_dim,
                                        out_channels=embedding_dim,
                                        init_weight_std=1.0,
                                        add_bias=embedding_bias,
                                        init_bias=0.0,
                                        lr_mul=embedding_lr_mul,
                                        activation_type='linear')
            self.var_mapping['embedding.weight'] = 'embed.weight'
            if self.embedding_bias:
                self.var_mapping['embedding.bias'] = 'embed.bias'

        if num_outputs is not None and not repeat_output:
            output_dim = output_dim * num_outputs
        for i in range(num_layers):
            in_channels = (input_dim if i == 0 else hidden_dim)
            out_channels = (output_dim if i == (num_layers - 1) else hidden_dim)
            self.add_module(f'dense{i}',
                            DenseLayer(in_channels=in_channels,
                                       out_channels=out_channels,
                                       init_weight_std=1.0,
                                       add_bias=True,
                                       init_bias=0.0,
                                       lr_mul=lr_mul,
                                       activation_type='lrelu'))
            self.var_mapping[f'dense{i}.weight'] = f'fc{i}.weight'
            self.var_mapping[f'dense{i}.bias'] = f'fc{i}.bias'

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
                 w_dim,
                 image_channels,
                 final_tanh,
                 output_scale,
                 num_layers,
                 num_critical,
                 fmaps_base,
                 fmaps_max,
                 kernel_size,
                 conv_clamp,
                 first_cutoff,
                 first_stopband,
                 last_stopband_rel,
                 margin_size,
                 filter_size,
                 act_upsampling,
                 use_radial_filter,
                 eps):
        super().__init__()

        self.resolution = resolution
        self.w_dim = w_dim
        self.image_channels = image_channels
        self.final_tanh = final_tanh
        self.output_scale = output_scale
        self.num_layers = num_layers
        self.num_critical = num_critical
        self.fmaps_base = fmaps_base
        self.fmaps_max = fmaps_max
        self.kernel_size = kernel_size
        self.conv_clamp = conv_clamp
        self.first_cutoff = first_cutoff
        self.first_stopband = first_stopband
        self.last_stopband_rel = last_stopband_rel
        self.margin_size = margin_size
        self.filter_size = filter_size
        self.act_upsampling = act_upsampling
        self.use_radial_filter = use_radial_filter
        self.eps = eps

        self.var_mapping = {}

        # Get layer settings.
        last_cutoff = resolution / 2
        last_stopband = last_cutoff * last_stopband_rel
        layer_indices = np.arange(num_layers + 1)
        exponents = np.minimum(layer_indices / (num_layers - num_critical), 1)
        cutoffs = first_cutoff * (last_cutoff / first_cutoff) ** exponents
        stopbands = (
            first_stopband * (last_stopband / first_stopband) ** exponents)
        sampling_rates = np.exp2(np.ceil(np.log2(
            np.minimum(stopbands * 2, self.resolution))))
        sampling_rates = np.int64(sampling_rates)
        half_widths = np.maximum(stopbands, sampling_rates / 2) - cutoffs
        sizes = sampling_rates + margin_size * 2
        sizes[-2:] = resolution
        sizes = np.int64(sizes)
        channels = np.rint(np.minimum((fmaps_base / 2) / cutoffs, fmaps_max))
        channels[-1] = image_channels
        channels = np.int64(channels)

        self.cutoffs = cutoffs
        self.stopbands = stopbands
        self.sampling_rates = sampling_rates
        self.half_widths = half_widths
        self.sizes = sizes
        self.channels = channels

        # Input layer, with positional encoding.
        self.early_layer = InputLayer(w_dim=w_dim,
                                      channels=channels[0],
                                      size=sizes[0],
                                      sampling_rate=sampling_rates[0],
                                      cutoff=cutoffs[0])
        self.var_mapping['early_layer.weight'] = 'input.weight'
        self.var_mapping['early_layer.affine.weight'] = 'input.affine.weight'
        self.var_mapping['early_layer.affine.bias'] = 'input.affine.bias'
        self.var_mapping['early_layer.transform'] = 'input.transform'
        self.var_mapping['early_layer.frequency'] = 'input.freqs'
        self.var_mapping['early_layer.phase'] = 'input.phases'

        # Convolutional layers.
        for idx in range(num_layers + 1):
            # Position related settings.
            if idx < num_layers:
                kernel_size = self.kernel_size
                demodulate = True
                act_upsampling = self.act_upsampling
            else:  # ToRGB layer.
                kernel_size = 1
                demodulate = False
                act_upsampling = 1
            if idx < num_layers - num_critical:  # Non-critical sampling.
                use_radial_filter = self.use_radial_filter
            else:  # Critical sampling.
                use_radial_filter = False

            prev_idx = max(idx - 1, 0)
            layer_name = f'layer{idx}'
            official_layer_name = f'L{idx}_{sizes[idx]}_{channels[idx]}'
            self.add_module(
                layer_name,
                SynthesisLayer(in_channels=channels[prev_idx],
                               out_channels=channels[idx],
                               w_dim=w_dim,
                               kernel_size=kernel_size,
                               demodulate=demodulate,
                               eps=eps,
                               conv_clamp=conv_clamp,
                               in_size=sizes[prev_idx],
                               out_size=sizes[idx],
                               in_sampling_rate=sampling_rates[prev_idx],
                               out_sampling_rate=sampling_rates[idx],
                               in_cutoff=cutoffs[prev_idx],
                               out_cutoff=cutoffs[idx],
                               in_half_width=half_widths[prev_idx],
                               out_half_width=half_widths[idx],
                               filter_size=filter_size,
                               use_radial_filter=use_radial_filter,
                               act_upsampling=act_upsampling))

            self.var_mapping[f'{layer_name}.magnitude_ema'] = (
                f'{official_layer_name}.magnitude_ema')
            self.var_mapping[f'{layer_name}.conv.weight'] = (
                f'{official_layer_name}.weight')
            self.var_mapping[f'{layer_name}.conv.style.weight'] = (
                f'{official_layer_name}.affine.weight')
            self.var_mapping[f'{layer_name}.conv.style.bias'] = (
                f'{official_layer_name}.affine.bias')
            self.var_mapping[f'{layer_name}.filter.bias'] = (
                f'{official_layer_name}.bias')
            if idx < num_layers:  # ToRGB layer does not need filters.
                self.var_mapping[f'{layer_name}.filter.up_filter'] = (
                    f'{official_layer_name}.up_filter')
                self.var_mapping[f'{layer_name}.filter.down_filter'] = (
                    f'{official_layer_name}.down_filter')

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
                magnitude_moving_decay=0.999,
                update_ema=False,
                fp16_res=None,
                impl='cuda'):
        results = {'wp': wp}

        x = self.early_layer(wp[:, 0])
        for idx, sampling_rate in enumerate(self.sampling_rates):
            if fp16_res is not None and sampling_rate >= fp16_res:
                x = x.to(torch.float16)
            layer = getattr(self, f'layer{idx}')
            x, style = layer(x, wp[:, idx + 1],
                             magnitude_moving_decay=magnitude_moving_decay,
                             update_ema=update_ema,
                             impl=impl)
            results[f'style{idx}'] = style

        if self.output_scale != 1:
            x = x * self.output_scale
        x = x.to(torch.float32)
        if self.final_tanh:
            x = torch.tanh(x)
        results['image'] = x
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
    """Implements the input layer with positional encoding.

    Basically, this block outputs a feature map with shape
    `(channels, size, size)` based on the coordinate information.
    `sampling_rate` and `cutoff` are used to control the coordinate range and
    strength respectively.

    For a low-pass filter, `cutoff` is the same as the `bandwidth`.
    The initial frequency of the starting feature map is controlled by the
    positional encoding `sin(2 * pi * x)`, where
    `x = trans(coord) * frequency + phase`. We would like to introduce rich
    information (i.e. frequencies), but keep all frequencies lower than
    stopband, which is `sampling_rate / 2`.

    Besides, this layer also supports learning a transformation from the latent
    code w, and providing a customized transformation for inference. Please
    use the buffer `transform`.

    NOTE: `size` is different from `sampling_rate`. `sampling_rate` is the
    actual size of the current stage, which determines the maximum frequency
    that the feature maps can hold. `size` is the actual height and width of the
    current feature map, including the extended border.
    """

    def __init__(self, w_dim, channels, size, sampling_rate, cutoff):
        super().__init__()

        self.w_dim = w_dim
        self.channels = channels
        self.size = size
        self.sampling_rate = sampling_rate
        self.cutoff = cutoff

        # Coordinate of the entire feature map, with resolution (size, size).
        # The coordinate range for the central (sampling_rate, sampling_rate)
        # region is set as (-0.0, 0.5), which extends to the remaining region.
        theta = torch.eye(2, 3)
        theta[0, 0] = 0.5 / sampling_rate * size
        theta[1, 1] = 0.5 / sampling_rate * size
        grid = F.affine_grid(theta=theta.unsqueeze(0),
                             size=(1, 1, size, size),
                             align_corners=False)
        self.register_buffer('grid', grid)

        # Draw random frequency from a uniform 2D disc for each channel
        # regarding X and Y dimension. And also draw a random phase for each
        # channel. Accordingly, each channel has three pre-defined parameters,
        # which are X-frequency, Y-frequency, and phase.
        frequency = torch.randn(channels, 2)
        radius = frequency.square().sum(dim=1, keepdim=True).sqrt()
        frequency = frequency / (radius * radius.square().exp().pow(0.25))
        frequency = frequency * cutoff
        self.register_buffer('frequency', frequency)
        phase = torch.rand(channels) - 0.5
        self.register_buffer('phase', phase)

        # This layer is used to map the latent code w to transform factors,
        # with order: cos(angle), sin(angle), transpose_x, transpose_y.
        self.affine = DenseLayer(in_channels=w_dim,
                                 out_channels=4,
                                 init_weight_std=0.0,
                                 add_bias=True,
                                 init_bias=(1, 0, 0, 0),
                                 lr_mul=1.0,
                                 activation_type='linear')

        # It is possible to use this buffer to customize the transform of the
        # output synthesis.
        self.register_buffer('transform', torch.eye(3))

        # Use 1x1 conv to convert positional encoding to features.
        self.weight = nn.Parameter(torch.randn(channels, channels))
        self.weight_scale = 1 / np.sqrt(channels)

    def extra_repr(self):
        return (f'channels={self.channels}, '
                f'size={self.size}, '
                f'sampling_rate={self.sampling_rate}, '
                f'cutoff={self.cutoff:.3f}, ')

    def forward(self, w):
        batch = w.shape[0]

        # Get transformation matrix.
        # Factor controlled by latent code.
        transformation_factor = self.affine(w)
        # Ensure the range of cosine and sine value (first two dimension).
        _norm = transformation_factor[:, :2].norm(dim=1, keepdim=True)
        transformation_factor = transformation_factor / _norm
        # Rotation.
        rotation = torch.eye(3, device=w.device).unsqueeze(0)
        rotation = rotation.repeat((batch, 1, 1))
        rotation[:, 0, 0] = transformation_factor[:, 0]
        rotation[:, 0, 1] = -transformation_factor[:, 1]
        rotation[:, 1, 0] = transformation_factor[:, 1]
        rotation[:, 1, 1] = transformation_factor[:, 0]
        # Translation.
        translation = torch.eye(3, device=w.device).unsqueeze(0)
        translation = translation.repeat((batch, 1, 1))
        translation[:, 0, 2] = -transformation_factor[:, 2]
        translation[:, 1, 2] = -transformation_factor[:, 3]
        # Customized transformation.
        transform = rotation @ translation @ self.transform.unsqueeze(0)

        # Transform frequency and shift, which is equivalent to transforming
        # the coordinate. For example, given a coordinate, X, we would like to
        # first transform it with the rotation matrix, R, and the translation
        # matrix, T, as X' = RX + T. Then, we will apply frequency, f, and
        # phase, p, with sin(2 * pi * (fX' + p)). Natively, we have
        # fX' + p = f(RX + T) + p = (fR)X + (fT + p)
        frequency = self.frequency.unsqueeze(0) @ transform[:, :2, :2]  # [NC2]
        phase = self.frequency.unsqueeze(0) @ transform[:, :2, 2:]      # [NC]
        phase = phase.squeeze(2) + self.phase.unsqueeze(0)              # [NC]

        # Positional encoding.
        x = self.grid                                                # [NHW2]
        x = x.unsqueeze(3)                                           # [NHW12]
        x = x @ frequency.transpose(1, 2).unsqueeze(1).unsqueeze(2)  # [NHW1C]
        x = x.squeeze(3)                                             # [NHWC]
        x = x + phase.unsqueeze(1).unsqueeze(2)                      # [NHWC]
        x = torch.sin(2 * np.pi * x)                                 # [NHWC]

        # Dampen out-of-band frequency that may be introduced by the customized
        # transform `self.transform`.
        frequency_norm = frequency.norm(dim=2)
        stopband = self.sampling_rate / 2
        factor = (frequency_norm - self.cutoff) / (stopband - self.cutoff)
        amplitude = (1 - factor).clamp(0, 1)         # [NC]
        x = x * amplitude.unsqueeze(1).unsqueeze(2)  # [NHWC]

        # Project positional encoding to features.
        weight = self.weight * self.weight_scale
        x = x @ weight.t()

        return x.permute(0, 3, 1, 2).contiguous()


class SynthesisLayer(nn.Module):
    """Implements the synthesis layer.

    Each synthesis layer (including ToRGB layer) consists of a
    `ModulateConvLayer` and a `FilteringActLayer`. Besides, this layer will
    trace the magnitude (norm) of the input feature map, and update the
    statistic with `magnitude_moving_decay`.
    """

    def __init__(self,
                 # Settings for modulated convolution.
                 in_channels,
                 out_channels,
                 w_dim,
                 kernel_size,
                 demodulate,
                 eps,
                 conv_clamp,
                 # Settings for filtering activation.
                 in_size,
                 out_size,
                 in_sampling_rate,
                 out_sampling_rate,
                 in_cutoff,
                 out_cutoff,
                 in_half_width,
                 out_half_width,
                 filter_size,
                 use_radial_filter,
                 act_upsampling):
        """Initializes with layer settings.

        Args:
            in_channels: Number of channels of the input tensor.
            out_channels: Number of channels of the output tensor.
            w_dim: Dimension of W space for style modulation.
            kernel_size: Size of the convolutional kernels.
            demodulate: Whether to perform style demodulation.
            eps: A small value to avoid divide overflow.
            conv_clamp: A threshold to clamp the output of convolution layers to
                avoid overflow under FP16 training.
            in_size: Size of the input feature map, i.e., height and width.
            out_size: Size of the output feature map, i.e., height and width.
            in_sampling_rate: Sampling rate of the input feature map. Different
                from `in_size` that includes extended border, this field
                controls the actual maximum frequency that can be represented
                by the feature map.
            out_sampling_rate: Sampling rate of the output feature map.
            in_cutoff: Cutoff frequency of the input feature map.
            out_cutoff: Cutoff frequency of the output feature map.
            in_half_width: Half-width of the transition band of the input
                feature map.
            out_half_width: Half-width of the transition band of the output
                feature map.
            filter_size: Size of the filter used in this layer.
            use_radial_filter: Whether to use radial filter.
            act_upsampling: Upsampling factor used before the activation.
                `1` means do not wrap upsampling and downsampling around the
                activation.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.eps = eps
        self.conv_clamp = conv_clamp

        self.in_size = in_size
        self.out_size = out_size
        self.in_sampling_rate = in_sampling_rate
        self.out_sampling_rate = out_sampling_rate
        self.in_cutoff = in_cutoff
        self.out_cutoff = out_cutoff
        self.in_half_width = in_half_width
        self.out_half_width = out_half_width
        self.filter_size = filter_size
        self.use_radial_filter = use_radial_filter
        self.act_upsampling = act_upsampling

        self.conv = ModulateConvLayer(in_channels=in_channels,
                                      out_channels=out_channels,
                                      w_dim=w_dim,
                                      kernel_size=kernel_size,
                                      demodulate=demodulate,
                                      eps=eps)
        self.register_buffer('magnitude_ema', torch.ones(()))
        self.filter = FilteringActLayer(out_channels=out_channels,
                                        in_size=in_size,
                                        out_size=out_size,
                                        in_sampling_rate=in_sampling_rate,
                                        out_sampling_rate=out_sampling_rate,
                                        in_cutoff=in_cutoff,
                                        out_cutoff=out_cutoff,
                                        in_half_width=in_half_width,
                                        out_half_width=out_half_width,
                                        filter_size=filter_size,
                                        use_radial_filter=use_radial_filter,
                                        conv_padding=self.conv.padding,
                                        act_upsampling=act_upsampling)

    def extra_repr(self):
        return f'conv_clamp={self.conv_clamp}'

    def forward(self,
                x,
                w,
                magnitude_moving_decay=0.999,
                update_ema=False,
                impl='cuda'):
        if self.training and update_ema and magnitude_moving_decay is not None:
            magnitude = x.detach().to(torch.float32).square().mean()
            self.magnitude_ema.copy_(
                magnitude.lerp(self.magnitude_ema, magnitude_moving_decay))

        input_gain = self.magnitude_ema.rsqrt()
        x, style = self.conv(x, w, gain=input_gain, impl=impl)
        if self.act_upsampling > 1:
            x = self.filter(x, np.sqrt(2), 0.2, self.conv_clamp, impl=impl)
        else:
            x = self.filter(x, 1, 1, self.conv_clamp, impl=impl)

        return x, style


class ModulateConvLayer(nn.Module):
    """Implements the convolutional layer with style modulation.

    Different from the one introduced in StyleGAN2, this layer has following
    changes:

    (1) fusing `conv` and `style modulation` into one op by default
    (2) NOT adding a noise onto the output feature map.
    (3) NOT activating the feature map, which is moved to `FilteringActLayer`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 w_dim,
                 kernel_size,
                 demodulate,
                 eps):
        """Initializes with layer settings.

        Args:
            in_channels: Number of channels of the input tensor.
            out_channels: Number of channels of the output tensor.
            w_dim: Dimension of W space for style modulation.
            kernel_size: Size of the convolutional kernels.
            demodulate: Whether to perform style demodulation.
            eps: A small value to avoid divide overflow.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.eps = eps

        self.space_of_latent = 'W'

        # Set up weight.
        weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.randn(*weight_shape))
        self.wscale = 1.0 / np.sqrt(kernel_size * kernel_size * in_channels)
        self.padding = kernel_size - 1

        # Set up style.
        self.style = DenseLayer(in_channels=w_dim,
                                out_channels=in_channels,
                                init_weight_std=1.0,
                                add_bias=True,
                                init_bias=1.0,
                                lr_mul=1.0,
                                activation_type='linear')

    def extra_repr(self):
        return (f'in_ch={self.in_channels}, '
                f'out_ch={self.out_channels}, '
                f'ksize={self.kernel_size}, '
                f'demodulate={self.demodulate}')

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

    def forward(self, x, w, gain=None, impl='cuda'):
        dtype = x.dtype
        N, C, H, W = x.shape

        # Affine on `w`.
        style = self.forward_style(w, impl=impl)
        if not self.demodulate:
            _style = style * self.wscale  # Equivalent to scaling weight.
        else:
            _style = style

        weight = self.weight
        out_ch, in_ch, kh, kw = weight.shape
        assert in_ch == C

        # Pre-normalize inputs.
        if self.demodulate:
            weight = (weight *
                      weight.square().mean(dim=(1, 2, 3), keepdim=True).rsqrt())
            _style = _style * _style.square().mean().rsqrt()

        weight = weight.unsqueeze(0)
        weight = weight * _style.reshape(N, 1, in_ch, 1, 1)  # modulation
        if self.demodulate:
            decoef = (weight.square().sum(dim=(2, 3, 4)) + self.eps).rsqrt()
            weight = weight * decoef.reshape(N, out_ch, 1, 1, 1)  # demodulation

        if gain is not None:
            gain = gain.expand(N, in_ch)
            weight = weight * gain.reshape(N, 1, in_ch, 1, 1)

        # Fuse `conv` and `style modulation` as one op, using group convolution.
        x = x.reshape(1, N * in_ch, H, W)
        w = weight.reshape(N * out_ch, in_ch, kh, kw).to(dtype)
        x = conv2d_gradfix.conv2d(
            x, w, padding=self.padding, groups=N, impl=impl)
        x = x.reshape(N, out_ch, x.shape[2], x.shape[3])

        assert x.dtype == dtype
        assert style.dtype == torch.float32
        return x, style


class FilteringActLayer(nn.Module):
    """Implements the activation, wrapped with upsampling and downsampling.

    Basically, this layer executes the following operations in order:

    (1) Apply bias.
    (2) Upsample the feature map to increase sampling rate.
    (3) Apply non-linearity as activation.
    (4) Downsample the feature map to target size.

    This layer is mostly borrowed from the official implementation:

    https://github.com/NVlabs/stylegan3/blob/main/training/networks_stylegan3.py
    """

    def __init__(self,
                 out_channels,
                 in_size,
                 out_size,
                 in_sampling_rate,
                 out_sampling_rate,
                 in_cutoff,
                 out_cutoff,
                 in_half_width,
                 out_half_width,
                 filter_size,
                 use_radial_filter,
                 conv_padding,
                 act_upsampling):
        """Initializes with layer settings.

        Args:
            out_channels: Number of output channels, which is used for `bias`.
            in_size: Size of the input feature map, i.e., height and width.
            out_size: Size of the output feature map, i.e., height and width.
            in_sampling_rate: Sampling rate of the input feature map. Different
                from `in_size` that includes extended border, this field
                controls the actual maximum frequency that can be represented
                by the feature map.
            out_sampling_rate: Sampling rate of the output feature map.
            in_cutoff: Cutoff frequency of the input feature map.
            out_cutoff: Cutoff frequency of the output feature map.
            in_half_width: Half-width of the transition band of the input
                feature map.
            out_half_width: Half-width of the transition band of the output
                feature map.
            filter_size: Size of the filter used in this layer.
            use_radial_filter: Whether to use radial filter.
            conv_padding: The padding used in the previous convolutional layer.
            act_upsampling: Upsampling factor used before the activation.
                `1` means do not wrap upsampling and downsampling around the
                activation.
        """
        super().__init__()

        self.out_channels = out_channels
        self.in_size = in_size
        self.out_size = out_size
        self.in_sampling_rate = in_sampling_rate
        self.out_sampling_rate = out_sampling_rate
        self.in_cutoff = in_cutoff
        self.out_cutoff = out_cutoff
        self.in_half_width = in_half_width
        self.out_half_width = out_half_width
        self.filter_size = filter_size
        self.use_radial_filter = use_radial_filter
        self.conv_padding = conv_padding
        self.act_upsampling = act_upsampling

        # Define bias.
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # This sampling rate describes the upsampled feature map before
        # activation.
        temp_sampling_rate = max(in_sampling_rate, out_sampling_rate)
        temp_sampling_rate = temp_sampling_rate * act_upsampling

        # Design upsampling filter.
        up_factor = int(np.rint(temp_sampling_rate / in_sampling_rate))
        assert in_sampling_rate * up_factor == temp_sampling_rate
        if up_factor > 1:
            self.up_factor = up_factor
            self.up_taps = filter_size * up_factor
        else:
            self.up_factor = 1
            self.up_taps = 1  # No filtering.
        self.register_buffer(
            'up_filter',
            self.design_lowpass_filter(numtaps=self.up_taps,
                                       cutoff=in_cutoff,
                                       width=in_half_width * 2,
                                       fs=temp_sampling_rate,
                                       radial=False))

        # Design downsampling filter.
        down_factor = int(np.rint(temp_sampling_rate / out_sampling_rate))
        assert out_sampling_rate * down_factor == temp_sampling_rate
        if down_factor > 1:
            self.down_factor = down_factor
            self.down_taps = filter_size * down_factor
        else:
            self.down_factor = 1
            self.down_taps = 1  # No filtering.
        self.register_buffer(
            'down_filter',
            self.design_lowpass_filter(numtaps=self.down_taps,
                                       cutoff=out_cutoff,
                                       width=out_half_width * 2,
                                       fs=temp_sampling_rate,
                                       radial=use_radial_filter))

        # Compute padding.
        # Desired output size before downsampling.
        pad_total = (out_size - 1) * self.down_factor + 1
        # Input size after upsampling.
        pad_total = pad_total - (in_size + conv_padding) * self.up_factor
        # Size reduction caused by the filters.
        pad_total = pad_total + self.up_taps + self.down_taps - 2
        # Shift sample locations according to the symmetric interpretation.
        pad_lo = (pad_total + self.up_factor) // 2
        pad_hi = pad_total - pad_lo
        self.padding = list(map(int, (pad_lo, pad_hi, pad_lo, pad_hi)))

    def extra_repr(self):
        return (f'in_size={self.in_size}, '
                f'out_size={self.out_size}, '
                f'in_srate={self.in_sampling_rate}, '
                f'out_srate={self.out_sampling_rate}, '
                f'in_cutoff={self.in_cutoff:.3f}, '
                f'out_cutoff={self.out_cutoff:.3f}, '
                f'in_half_width={self.in_half_width:.3f}, '
                f'out_half_width={self.out_half_width:.3f}, '
                f'up_factor={self.up_factor}, '
                f'up_taps={self.up_taps}, '
                f'down_factor={self.down_factor}, '
                f'down_taps={self.down_taps}, '
                f'filter_size={self.filter_size}, '
                f'radial_filter={self.use_radial_filter}, '
                f'conv_padding={self.conv_padding}, '
                f'act_upsampling={self.act_upsampling}')

    @staticmethod
    def design_lowpass_filter(numtaps, cutoff, width, fs, radial=False):
        """Designs a low-pass filter.

        Args:
            numtaps: Length of the filter (number of coefficients, i.e., the
                filter order + 1).
            cutoff: Cutoff frequency of the output filter.
            width: Width of the transition region.
            fs: Sampling frequency.
            radial: Whether to use radially symmetric jinc-based filter.
                (default: False)
        """
        if numtaps == 1:
            return None

        assert numtaps > 1

        if not radial:  # Separable Kaiser low-pass filter.
            f = scipy.signal.firwin(numtaps=numtaps,
                                    cutoff=cutoff,
                                    width=width,
                                    fs=fs)
        else:  # Radially symmetric jinc-based filter.
            x = (np.arange(numtaps) - (numtaps - 1) / 2) / fs
            r = np.hypot(*np.meshgrid(x, x))
            f = scipy.special.j1(2 * cutoff * (np.pi * r)) / (np.pi * r)
            beta = scipy.signal.kaiser_beta(
                scipy.signal.kaiser_atten(numtaps, width / (fs / 2)))
            w = np.kaiser(numtaps, beta)
            f = f * np.outer(w, w)
            f = f / np.sum(f)
        return torch.as_tensor(f, dtype=torch.float32)

    def forward(self, x, gain, slope, clamp, impl='cuda'):
        dtype = x.dtype

        x = filtered_lrelu.filtered_lrelu(x=x,
                                          fu=self.up_filter,
                                          fd=self.down_filter,
                                          b=self.bias.to(dtype),
                                          up=self.up_factor,
                                          down=self.down_factor,
                                          padding=self.padding,
                                          gain=gain,
                                          slope=slope,
                                          clamp=clamp,
                                          impl=impl)

        assert x.dtype == dtype
        return x


class DenseLayer(nn.Module):
    """Implements the dense layer."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 init_weight_std,
                 add_bias,
                 init_bias,
                 lr_mul,
                 activation_type):
        """Initializes with layer settings.

        Args:
            in_channels: Number of channels of the input tensor.
            out_channels: Number of channels of the output tensor.
            init_weight_std: The initial standard deviation of weight.
            add_bias: Whether to add bias onto the fully-connected result.
            init_bias: The initial bias value before training.
            lr_mul: Learning multiplier for both weight and bias.
            activation_type: Type of activation.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_weight_std = init_weight_std
        self.add_bias = add_bias
        self.init_bias = init_bias
        self.lr_mul = lr_mul
        self.activation_type = activation_type

        weight_shape = (out_channels, in_channels)
        self.weight = nn.Parameter(
            torch.randn(*weight_shape) * init_weight_std / lr_mul)
        self.wscale = lr_mul / np.sqrt(in_channels)

        if add_bias:
            init_bias = np.float32(np.float32(init_bias) / lr_mul)
            if isinstance(init_bias, np.float32):
                self.bias = nn.Parameter(torch.full([out_channels], init_bias))
            else:
                assert isinstance(init_bias, np.ndarray)
                self.bias = nn.Parameter(torch.from_numpy(init_bias))
            self.bscale = lr_mul
        else:
            self.bias = None

    def extra_repr(self):
        return (f'in_ch={self.in_channels}, '
                f'out_ch={self.out_channels}, '
                f'init_weight_std={self.init_weight_std}, '
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

# pylint: enable=missing-function-docstring
