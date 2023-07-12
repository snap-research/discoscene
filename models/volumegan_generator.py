# python3.7

"""Contains the implementation of generator described in VolumeGAN.

Paper: https://arxiv.org/pdf/2112.10759.pdf
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from third_party.stylegan2_official_ops import fma
from third_party.stylegan2_official_ops import bias_act
from third_party.stylegan2_official_ops import upfirdn2d
from third_party.stylegan2_official_ops import conv2d_gradfix

from .utils.ops import all_gather
from .rendering import PointsSampling, HierarchicalSampling, Renderer, interpolate_feature
from .stylegan2_generator import ModulateConvLayer, ConvLayer, DenseLayer

__all__ = ['VolumeGANGenerator']

class VolumeGANGenerator(nn.Module):
    """Defines the generator network in VoumeGAN.

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

    (1) label_size: Size of the additional label for conditional generation.
        (default: 0)
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
        `resnet`. (default: `resnet`)
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
    (8) impl: Implementation mode of some particular ops, e.g., `filtering`,
        `bias_act`, etc. `cuda` means using the official CUDA implementation
        from StyleGAN2, while `ref` means using the native torch ops.
        (default: `cuda`)
    (9) fp16_res: Layers at resolution higher than (or equal to) this field will
        use `float16` precision for computation. This is merely used for
        acceleration. If set as `None`, all layers will use `float32` by
        default. (default: None)
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
                 label_size=0,
                 embedding_dim=512,
                 embedding_bias=True,
                 embedding_use_wscale=True,
                 embedding_wscale_gian=1.0,
                 embedding_lr_mul=1.0,
                 normalize_embedding=True,
                 normalize_embedding_latent=False,
                 # Settings for synthesis network.
                 resolution=-1,
                 nerf_res=32,
                 image_channels=3,
                 final_tanh=False,
                 demodulate=True,
                 use_wscale=True,
                 wscale_gain=1.0,
                 lr_mul=1.0,
                 noise_type='spatial',
                 fmaps_base=32 << 10,
                 fmaps_max=512,
                 filter_kernel=(1, 3, 3, 1),
                 conv_clamp=None,
                 eps=1e-8,
                 rgb_init_res_out=True,
                 # Setting for NeRF synthesis network
                 fv_cfg=dict(feat_res=32,
                             init_res=4,
                             base_channels=256,
                             output_channels=32,
                             w_dim=512),
                 embed_cfg=dict(input_dim=3,
                                max_freq_log2=10-1,
                                N_freqs=10),
                 fg_cfg=dict(num_layers=4,
                              hidden_dim=256,
                              activation_type='lrelu'),
                 bg_cfg=None,
                 out_dim=512,
                 # Setting for point sampling
                 ps_cfg=dict(),
                 # Setting for hierarchical sampling
                 hs_cfg=dict(),
                 # Setting for volume rendering
                 vr_cfg=dict()
                 ):
        """Initializes with basic settings.

        Raises:
            ValueError: If the `resolution` is not supported, or `architecture`
                is not supported.
        """
        super().__init__()

        self.z_dim = z_dim
        self.w_dim = w_dim
        self.repeat_w = repeat_w
        self.normalize_z = normalize_z
        self.mapping_layers = mapping_layers
        self.mapping_fmaps = mapping_fmaps
        self.mapping_use_wscale = mapping_use_wscale
        self.mapping_wscale_gain = mapping_wscale_gain
        self.mapping_lr_mul = mapping_lr_mul

        self.label_size = label_size
        self.label_dim = label_size
        self.embedding_dim = embedding_dim
        self.embedding_bias = embedding_bias
        self.embedding_use_wscale = embedding_use_wscale
        self.embedding_wscale_gain = embedding_wscale_gian
        self.embedding_lr_mul = embedding_lr_mul
        self.normalize_embedding = normalize_embedding
        self.normalize_embedding_latent = normalize_embedding_latent

        self.resolution = resolution
        self.nerf_res = nerf_res
        self.image_channels = image_channels
        self.final_tanh = final_tanh
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
        self.num_nerf_layers = fg_cfg['num_layers']
        self.num_cnn_layers = int(np.log2(resolution // nerf_res * 2)) * 2
        self.num_layers = self.num_nerf_layers + self.num_cnn_layers
      
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
            label_size=label_size,
            embedding_dim=embedding_dim,
            embedding_bias=embedding_bias,
            embedding_use_wscale=embedding_use_wscale,
            embedding_wscale_gian=embedding_wscale_gian,
            embedding_lr_mul=embedding_lr_mul,
            normalize_embedding=normalize_embedding,
            normalize_embedding_latent=normalize_embedding_latent,
            eps=eps)

        self.nerfmlp = NeRFSynthesisNetwork(
            fv_cfg=fv_cfg,
            embed_cfg=embed_cfg,
            fg_cfg=fg_cfg,
            bg_cfg=bg_cfg,
            out_dim=out_dim,
        )

        # This is used for truncation trick.
        if self.repeat_w:
            self.register_buffer('w_avg', torch.zeros(w_dim))
        else:
            self.register_buffer('w_avg', torch.zeros(self.num_layers * w_dim))

        self.synthesis = SynthesisNetwork(resolution=resolution,
                                          init_res=nerf_res,
                                          w_dim=w_dim,
                                          image_channels=image_channels,
                                          final_tanh=final_tanh,
                                          demodulate=demodulate,
                                          use_wscale=use_wscale,
                                          wscale_gain=wscale_gain,
                                          lr_mul=lr_mul,
                                          noise_type=noise_type,
                                          fmaps_base=fmaps_base,
                                          filter_kernel=filter_kernel,
                                          fmaps_max=fmaps_max,
                                          conv_clamp=conv_clamp,
                                          eps=eps,
                                          rgb_init_res_out=rgb_init_res_out)

        self.pointsampler = PointsSampling(**ps_cfg)
        self.hierachicalsampler = HierarchicalSampling(**hs_cfg)
        self.volumerenderer = Renderer(**vr_cfg)

    def set_space_of_latent(self, space_of_latent):
        """Sets the space to which the latent code belong.

        See `SynthesisNetwork` for more details.
        """
        self.synthesis.set_space_of_latent(space_of_latent)

    def nerf_synthesis(self, 
                       w, 
                       noise_std=0,
                       ps_kwargs=dict()):
        ps_results = self.pointsampler(batch_size=w.shape[0],
                                       resolution=self.nerf_res,
                                       **ps_kwargs)
        nerf_synthesis_results = self.nerfmlp(wp=w,
                                              pts=ps_results['pts'],
                                              dirs=ps_results['ray_dirs'])
        hs_results = self.hierachicalsampler(coarse_rgbs=nerf_synthesis_results['rgb'],
                                            coarse_sigmas=nerf_synthesis_results['sigma'],
                                            pts_z=ps_results['pts_z'],
                                            ray_origins=ps_results['ray_origins'],
                                            ray_dirs=ps_results['ray_dirs'],
                                            noise_std=noise_std)
        fine_nerf_synthesis_results = self.nerfmlp(wp=w,
                                                pts=hs_results['pts'],
                                                dirs=hs_results['ray_dirs'])
        # Concat coarse and fine results
        rgbs = torch.cat((fine_nerf_synthesis_results['rgb'], nerf_synthesis_results['rgb']), dim=-2)
        sigmas = torch.cat((fine_nerf_synthesis_results['sigma'], nerf_synthesis_results['sigma']), dim=-2)
        pts_z_all = torch.cat((hs_results['pts_z'], ps_results['pts_z']), dim=-2)
        _, indices = torch.sort(pts_z_all, dim=-2)
        rgbs = torch.gather(rgbs, -2, indices.expand(-1,-1,-1,-1,rgbs.shape[-1]))
        sigmas = torch.gather(sigmas, -2, indices)
        pts_z_all = torch.gather(pts_z_all, -2, indices)
        # Volume Rendering
        render_results = self.volumerenderer(rgbs=rgbs,
                                            sigmas=sigmas,
                                            pts_z=pts_z_all,
                                            noise_std=noise_std)
        # nerf output
        nerf_feat = render_results['rgb'].permute(0, 3, 1, 2)
        return nerf_feat


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
                fused_modulate=False,
                impl='cuda',
                fp16_res=None,
                ps_kwargs=dict(),
                ):

        """Connects mapping network and synthesis network.

        This forward function will also update the average `w_code`, perform
        style mixing as a training regularizer, and do truncation trick, which
        is specially designed for inference.

        Concretely, the truncation trick acts as follows:

        For layers in range [0, truncation_layers), the truncated w-code is
        computed as

        w_new = w_avg + (w - w_avg) * truncation_psi

        To disable truncation, please set

        (1) truncation_psi = 1.0 (None) OR
        (2) truncation_layers = 0 (None)
        """

        mapping_results = self.mapping(z, label, impl=impl)
        w = mapping_results['w']
        lod = self.synthesis.lod.item() if lod is None else lod

        if self.training and w_moving_decay is not None:
            if sync_w_avg:
                batch_w_avg = all_gather(w.detach()).mean(dim=0)
            else:
                batch_w_avg = w.detach().mean(dim=0)
            self.w_avg.copy_(batch_w_avg.lerp(self.w_avg, w_moving_decay))

        wp = mapping_results['wp']

        if self.training and style_mixing_prob is not None:
            if np.random.uniform() < style_mixing_prob:
                new_z = torch.randn_like(z)
                new_wp = self.mapping(new_z, label, impl=impl)['wp']
                current_layers = self.num_layers
                if current_layers > self.num_nerf_layers:
                    mixing_cutoff = np.random.randint(self.num_nerf_layers, current_layers)
                    wp[:, mixing_cutoff:] = new_wp[:, mixing_cutoff:]

        if not self.training:
            trunc_psi = 1.0 if trunc_psi is None else trunc_psi
            trunc_layers = 0 if trunc_layers is None else trunc_layers
            if trunc_psi < 1.0 and trunc_layers > 0:
                w_avg = self.w_avg.reshape(1, -1, self.w_dim)[:, :trunc_layers]
                wp[:, :trunc_layers] = w_avg.lerp(
                    wp[:, :trunc_layers], trunc_psi)

        nerf_w = wp[:,:self.num_nerf_layers]
        cnn_w = wp[:,self.num_nerf_layers:]
        feature2d = self.nerf_synthesis(w=nerf_w,
                                        ps_kwargs=ps_kwargs)
        synthesis_results = self.synthesis(feature2d,
                                           cnn_w,
                                           lod=None,
                                           noise_mode=noise_mode,
                                           fused_modulate=fused_modulate,
                                           impl=impl,
                                           fp16_res=fp16_res)

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
                 label_size,
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
        self.label_size = label_size
        self.embedding_dim = embedding_dim
        self.embedding_bias = embedding_bias
        self.embedding_use_wscale = embedding_use_wscale
        self.embedding_lr_mul = embedding_lr_mul
        self.normalize_embedding = normalize_embedding
        self.normalize_embedding_latent = normalize_embedding_latent
        self.eps = eps


        self.norm = PixelNormLayer(dim=1, eps=eps)

        if self.label_size > 0:
            input_dim = input_dim + embedding_dim
            self.embedding = DenseLayer(in_channels=label_size,
                                        out_channels=embedding_dim,
                                        add_bias=embedding_bias,
                                        init_bias=0.0,
                                        use_wscale=embedding_use_wscale,
                                        wscale_gain=embedding_wscale_gian,
                                        lr_mul=embedding_lr_mul,
                                        activation_type='linear')

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

    def forward(self, z, label=None, impl='cuda'):
        if z.ndim != 2 or z.shape[1] != self.input_dim:
            raise ValueError(f'Input latent code should be with shape '
                             f'[batch_size, input_dim], where '
                             f'`input_dim` equals to {self.input_dim}!\n'
                             f'But `{z.shape}` is received!')
        if self.normalize_input:
            z = self.norm(z)

        if self.label_size > 0:
            if label is None:
                raise ValueError(f'Model requires an additional label '
                                 f'(with size {self.label_size}) as input, '
                                 f'but no label is received!')
            if label.ndim != 2 or label.shape != (z.shape[0], self.label_size):
                raise ValueError(f'Input label should be with shape '
                                 f'[batch_size, label_size], where '
                                 f'`batch_size` equals to that of '
                                 f'latent codes ({z.shape[0]}) and '
                                 f'`label_size` equals to {self.label_size}!\n'
                                 f'But `{label.shape}` is received!')
            embedding = self.embedding(label, impl=impl)
            if self.normalize_embedding:
                embedding = self.norm(embedding)
            w = torch.cat((z, embedding), dim=1)
        else:
            w = z

        if self.label_size > 0 and self.normalize_embedding_latent:
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
        if self.label_size > 0:
            results['embedding'] = embedding
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

    Basically, this block starts from a 3D const input, which is with shape
    `(channels, init_res, init_res, init_res)`.
    """

    def __init__(self, init_res, channels):
        super().__init__()
        self.const = nn.Parameter(torch.randn(1, channels, init_res, init_res))

    def forward(self, w, input=None, **_unused_kwargs):
        # NOTE: For runtime arguments compatibility, do not remove `**kwargs`.
        if input is None:
            x = self.const.repeat(w.shape[0], 1, 1, 1)
        else:
            x = input
        return x


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
                 demodulate,
                 use_wscale,
                 wscale_gain,
                 lr_mul,
                 noise_type,
                 fmaps_base,
                 fmaps_max,
                 filter_kernel,
                 conv_clamp,
                 eps,
                 rgb_init_res_out=False,
                 ):
        super().__init__()

        self.init_res = init_res
        self.init_res_log2 = int(np.log2(init_res))
        self.resolution = resolution
        self.final_res_log2 = int(np.log2(resolution))
        self.w_dim = w_dim
        self.image_channels = image_channels
        self.final_tanh = final_tanh
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
        self.rgb_init_res_out = rgb_init_res_out

        self.num_layers = (self.final_res_log2 - self.init_res_log2 + 1) * 2

        self.register_buffer('lod', torch.zeros(()))

        for res_log2 in range(self.init_res_log2, self.final_res_log2 + 1):
            res = 2 ** res_log2
            in_channels = self.get_nf(res // 2)
            out_channels = self.get_nf(res)
            block_idx = res_log2 - self.init_res_log2

            # Early layer.
            if res > init_res:
                layer_name = f'layer{2 * block_idx - 1}'
                self.add_module(layer_name,
                                ModulateConvLayer(in_channels=in_channels,
                                                  out_channels=out_channels,
                                                  resolution=res,
                                                  w_dim=w_dim,
                                                  kernel_size=1,
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
            if block_idx == 0:
                if self.rgb_init_res_out:
                    self.rgb_init_res= ConvLayer(in_channels=out_channels,
                                                  out_channels=image_channels,
                                                  kernel_size=1,
                                                  add_bias=True,
                                                  scale_factor=1,
                                                  filter_kernel=None,
                                                  use_wscale=use_wscale,
                                                  wscale_gain=wscale_gain,
                                                  lr_mul=lr_mul,
                                                  activation_type='linear',
                                                  conv_clamp=conv_clamp,
                                                  ) 
                continue
            # Second layer (kernel 1x1) without upsampling.
            layer_name = f'layer{2 * block_idx}'
            self.add_module(layer_name,
                            ModulateConvLayer(in_channels=out_channels,
                                              out_channels=out_channels,
                                              resolution=res,
                                              w_dim=w_dim,
                                              kernel_size=1,
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

            # Output convolution layer for each resolution (if needed).
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
             
        # Used for upsampling output images for each resolution block for sum.
        self.register_buffer(
            'filter', upfirdn2d.setup_filter(filter_kernel))

    def get_nf(self, res):
        """Gets number of feature maps according to current resolution."""
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
                x,
                wp,
                lod=None,
                noise_mode='const',
                fused_modulate=False,
                impl='cuda',
                fp16_res=None,
                nerf_out=False):
        lod = self.lod.item() if lod is None else lod

        results = {}

        # Cast to `torch.float16` if needed.
        if fp16_res is not None and self.init_res >= fp16_res:
            x = x.to(torch.float16)

        for res_log2 in range(self.init_res_log2, self.final_res_log2 + 1):
            cur_lod = self.final_res_log2 - res_log2
            block_idx = res_log2 - self.init_res_log2 
            
            layer_idxs = [2*block_idx-1, 2*block_idx] if block_idx > 0 else[2*block_idx, ]
            # determine forward until cur resolution
            if lod < cur_lod + 1:
                for layer_idx in layer_idxs:
                    if layer_idx == 0:
                        # image = x[:,:3]
                        if self.rgb_init_res_out:
                            cur_image = self.rgb_init_res(x, runtime_gain=1, impl=impl)
                        else:
                            cur_image = x[:,:3]
                        continue 
                    layer = getattr(self, f'layer{layer_idx}')
                    x, style = layer(x,
                                     wp[:, layer_idx],
                                     noise_mode=noise_mode,
                                     fused_modulate=fused_modulate,
                                     impl=impl,)
                    results[f'style{layer_idx}'] = style
                    if layer_idx % 2 == 0:
                        output_layer = getattr(self, f'output{layer_idx // 2}')
                        y, style = output_layer(x,
                                                wp[:, layer_idx + 1],
                                                fused_modulate=fused_modulate,
                                                impl=impl,
                                                )
                        results[f'output_style{layer_idx // 2}'] = style
                        if layer_idx == 0:
                            cur_image = y.to(torch.float32)
                        else:
                            if not nerf_out:
                                cur_image = y.to(torch.float32) + upfirdn2d.upsample2d(
                                    cur_image, self.filter, impl=impl)
                            else:
                                cur_image = y.to(torch.float32) + cur_image

                        # Cast to `torch.float16` if needed.
                        if layer_idx != self.num_layers - 2:
                            res = self.init_res * (2 ** (layer_idx // 2))
                            if fp16_res is not None and res * 2 >= fp16_res:
                                x = x.to(torch.float16)
                            else:
                                x = x.to(torch.float32) 

            # rgb interpolation
            if cur_lod - 1 < lod <= cur_lod:
                image = cur_image
            elif cur_lod < lod < cur_lod + 1:
                alpha = np.ceil(lod) - lod
                image = F.interpolate(image, scale_factor=2, mode='nearest')
                image = cur_image * alpha + image * (1 - alpha)
            elif lod >= cur_lod + 1:
                image = F.interpolate(image, scale_factor=2, mode='nearest')

        if self.final_tanh:
            image = torch.tanh(image)
        results['image'] = image
        return results

class Embedder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
        '''
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq; min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        '''
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input):
        '''
        :param input: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        '''
        assert (input.shape[-1] == self.input_dim)

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))
        out = torch.cat(out, dim=-1)

        assert (out.shape[-1] == self.out_dim)
        return out

class NeRFSynthesisNetwork(nn.Module):
    def __init__(self,
                 fv_cfg=dict(feat_res=32,
                             init_res=4,
                             base_channels=512//2,
                             output_channels=32,
                             w_dim=512),
                 embed_cfg=dict(input_dim=3,
                                max_freq_log2=10-1,
                                N_freqs=10),
                 fg_cfg=dict(num_layers=4,
                             hidden_dim=256,
                             activation_type='lrelu',
                             ),
                 bg_cfg=None,
                 out_dim=512,
                 ):
        super().__init__()
        self.fg_cfg = fg_cfg
        self.bg_cfg = bg_cfg

        self.fv = FeatureVolume(**fv_cfg)
        self.fv_cfg = fv_cfg

        self.fg_embedder = Embedder(**embed_cfg)

        input_dim = self.fg_embedder.out_dim + self.fv_cfg['output_channels']

        self.fg_mlps = self.build_mlp(input_dim=input_dim, **fg_cfg)
        self.fg_density = DenseLayer(in_channels=fg_cfg['hidden_dim'],
                                           out_channels=1,
                                           add_bias=True,
                                           init_bias=0.0,
                                           use_wscale=True,
                                           wscale_gain=1,
                                           lr_mul=1,
                                           activation_type='linear')
        self.fg_color = DenseLayer(in_channels=fg_cfg['hidden_dim'],
                                             out_channels=out_dim,
                                             add_bias=True,
                                             init_bias=0.0,
                                             use_wscale=True,
                                             wscale_gain=1,
                                             lr_mul=1,
                                             activation_type='linear')
        if self.bg_cfg:
            self.bg_embedder = Embedder(**bg_cfg.embed_cfg)
            input_dim = self.bg_embedder.out_dim
            self.bg_mlps = self.build_mlp(input_dim, **bg_cfg)
            self.bg_density = DenseLayer(in_channels=bg_cfg['hidden_dim'],
                                                   out_channels=1,
                                                   add_bias=True,
                                                   init_bias=0.0,
                                                   use_wscale=True,
                                                   wscale_gain=1,
                                                   lr_mul=1,
                                                   activation_type='linear')
            self.bg_color = DenseLayer(in_channels=bg_cfg['hidden_dim'],
                                                 out_channels=out_dim,
                                                 add_bias=True,
                                                 init_bias=0.0,
                                                 use_wscale=True,
                                                 wscale_gain=1,
                                                 lr_mul=1,
                                                 activation_type='linear')

    def init_weights(self,):
        pass

    def build_mlp(self, input_dim, num_layers, hidden_dim, activation_type, **kwargs):
        default_conv_cfg = dict(resolution=32,
                                w_dim=512,
                                kernel_size=1,
                                add_bias=True,
                                scale_factor=1,
                                filter_kernel=None,
                                demodulate=True,
                                use_wscale=True,
                                wscale_gain=1,
                                lr_mul=1,
                                noise_type='none',
                                conv_clamp=None,
                                eps=1e-8)
        mlp_list = nn.ModuleList()
        in_ch = input_dim
        out_ch = hidden_dim
        for _ in range(num_layers):
            mlp = ModulateConvLayer(in_channels=in_ch,
                                    out_channels=out_ch,
                                    activation_type=activation_type,
                                    **default_conv_cfg)
            mlp_list.append(mlp)
            in_ch = out_ch
            out_ch = hidden_dim

        return mlp_list


    def forward(self, wp, pts, dirs, fused_modulate=False, impl='cuda'):

        hi, wi = pts.shape[1:3]
        fg_pts = rearrange(pts, 'bs h w d c -> bs (h w) d c').contiguous()
        w = wp              
        # pts: bs, h*h, d, 3
        fg_pts_embed = self.fg_embedder(fg_pts)
        bs, nump, numd, c = fg_pts_embed.shape
        fg_pts_embed = rearrange(fg_pts_embed, 'bs nump numd c -> bs c (nump numd) 1').contiguous()
        x = fg_pts_embed

        # feature volume
        if w.ndim == 3:
            fvw = w[:, 0]
        else:
            fvw = w
        volume = self.fv(fvw)
        # interpolate features from feature volume
        # point features: batch_size, num_channel, num_point
        bounds = self.fv_cfg.get('bounds', [[-0.1886, -0.1671, -0.1956],
                               [0.1887, 0.1692, 0.1872]])
        bounds = torch.Tensor(bounds).to(pts)

        fg_pts_sam = rearrange(fg_pts, 'bs nump numd c -> bs (nump numd) c')
        input_f = interpolate_feature(fg_pts_sam, volume, bounds)
        input_f = rearrange(input_f, 'bs c numd -> bs c numd 1')
        x = torch.cat([input_f, x], dim=1)

        for mlp_idx, fg_mlp in enumerate(self.fg_mlps):
            if wp.ndim == 3:
                lw = wp[:, mlp_idx]
            else:
                lw = wp
            x, style = fg_mlp(x, lw, fused_modulate=fused_modulate, impl=impl)
        fg_feat = rearrange(x, 'bs c (nump numd) 1 -> (bs nump numd) c', numd=numd).contiguous()
        fg_density = self.fg_density(fg_feat)
        fg_color = self.fg_color(fg_feat)

        fg_density = rearrange(fg_density, '(bs h w numd) c -> bs h w numd c', h=hi, w=wi, numd=numd).contiguous()
        fg_color = rearrange(fg_color, '(bs h w numd) c -> bs h w numd c', h=hi, w=wi, numd=numd).contiguous()

        if self.bg_cfg is not None and bg_pts is not None:
            # inverted sphere parameterization
            r = torch.norm(bg_pts, dim=-1)
            bg_pts = bg_pts / r[..., None]
            bg_pts = torch.cat([bg_pts, 1 / r[..., None]], dim=-1)

            bg_pts_embed = self.bg_embedder(bg_pts)
            bs, nump, numd, c = bg_pts_embed.shape
            bg_pts_embed = rearrange(bg_pts_embed, 'bs nump numd c -> bs c (nump numd) 1').contiguous()
            x = bg_pts_embed
            for bg_mlp in self.bg_mlps:
                x, style = bg_mlp(x, w, fused_modulate=fused_modulate, impl=impl)
            bg_feat = rearrange(x, 'bs c (nump numd) 1 -> (bs nump numd) c', numd=numd).contiguous()
            bg_density = self.bg_density(bg_feat)
            bg_color = self.bg_color(bg_feat)

            bg_density = rearrange(bg_density, '(bs  nump numd) c -> bs nump numd c', nump=nump, numd=numd).contiguous()
            bg_color = rearrange(bg_color, '(bs  nump numd) c -> bs nump numd c', nump=nump, numd=numd).contiguous()
        else:
            bg_color = None
            bg_density = None

        results = {
            'sigma': fg_density,
            'rgb': fg_color,
        }
        return results



def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight,
                                      a=0.2,
                                      mode='fan_in',
                                      nonlinearity='leaky_relu')

class InstanceNormLayer(nn.Module):
    """Implements instance normalization layer."""
    def __init__(self, num_features, epsilon=1e-8, affine=False):
        super().__init__()
        self.eps = epsilon
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(1, num_features,1,1,1))
            self.bias = nn.Parameter(torch.Tensor(1, num_features,1,1,1))
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def forward(self, x, weight=None, bias=None):
        x = x - torch.mean(x, dim=[2, 3, 4], keepdim=True)
        norm = torch.sqrt(
            torch.mean(x**2, dim=[2, 3, 4], keepdim=True) + self.eps)
        x = x / norm
        isnot_input_none = weight is not None and bias is not None
        assert (isnot_input_none and not self.affine) or (not isnot_input_none and self.affine)
        if self.affine:
            x = x*self.weight + self.bias
        else:
            x = x*weight + bias
        return x

class UpsamplingLayer(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        if self.scale_factor <= 1:
            return x
        return F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')


class FeatureVolume(nn.Module):
    def __init__(
        self,
        feat_res=32,
        init_res=4,
        base_channels=256,
        output_channels=32,
        w_dim=512,
        **kwargs
    ):
        super().__init__()
        self.num_stages = int(np.log2(feat_res // init_res)) + 1

        self.const = nn.Parameter(
            torch.ones(1, base_channels, init_res, init_res, init_res))
        inplanes = base_channels
        outplanes = base_channels

        self.stage_channels = []
        for i in range(self.num_stages):
            conv = nn.Conv3d(inplanes,
                             outplanes,
                             kernel_size=(3, 3, 3),
                             padding=(1, 1, 1))
            self.stage_channels.append(outplanes)
            self.add_module(f'layer{i}', conv)
            instance_norm = InstanceNormLayer(num_features=outplanes, affine=False)

            self.add_module(f'instance_norm{i}', instance_norm)
            inplanes = outplanes
            outplanes = max(outplanes // 2, output_channels)
            if i == self.num_stages - 1:
                outplanes = output_channels

        self.mapping_network = nn.Linear(w_dim, sum(self.stage_channels) * 2)
        self.mapping_network.apply(kaiming_leaky_init)
        with torch.no_grad(): self.mapping_network.weight *= 0.25
        self.upsample = UpsamplingLayer()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, w, **kwargs):
        scale_shifts = self.mapping_network(w)
        scales = scale_shifts[..., :scale_shifts.shape[-1]//2]
        shifts = scale_shifts[..., scale_shifts.shape[-1]//2:]

        x = self.const.repeat(w.shape[0], 1, 1, 1, 1)
        for idx in range(self.num_stages):
            if idx != 0:
                x = self.upsample(x)
            conv_layer = self.__getattr__(f'layer{idx}')
            x = conv_layer(x)
            instance_norm = self.__getattr__(f'instance_norm{idx}')
            scale = scales[:, sum(self.stage_channels[:idx]):sum(self.stage_channels[:idx + 1])]
            shift = shifts[:, sum(self.stage_channels[:idx]):sum(self.stage_channels[:idx + 1])]
            scale = scale.view(scale.shape + (1, 1, 1))
            shift = shift.view(shift.shape + (1, 1, 1))
            x = instance_norm(x, weight=scale, bias=shift)
            x = self.lrelu(x)

        return x
