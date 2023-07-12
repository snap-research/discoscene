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

import math

from .utils.ops import all_gather
from .rendering import PointsBboxSampling, HierarchicalBboxSampling, RendererBbox
from utils.image_utils import save_image, load_image, resize_image
from third_party.stylegan2_official_ops import fma                                                                     
from third_party.stylegan2_official_ops import bias_act
from third_party.stylegan2_official_ops import upfirdn2d
from third_party.stylegan2_official_ops import conv2d_gradfix
from .stylegan2_generator import ModulateConvLayer, ConvLayer, DenseLayer

__all__ = ['PiGANBboxGenerator']

class PiGANBboxGenerator(nn.Module):
    """Defines the generator network in StyleGAN.

    NOTE: The synthesized images are with `RGB` channel order and pixel range
    [-1, 1].

    Settings for the mapping network:

    (1) z_space_dim: Dimension of the input latent space, Z. (default: 512)
    (2) w_space_dim: Dimension of the outout latent space, W. (default: 512)
    (3) label_size: Size of the additional label for conditional generation.
        (default: 0)
    (4）mapping_layers: Number of layers of the mapping network. (default: 8)
    (5) mapping_fmaps: Number of hidden channels of the mapping network.
        (default: 512)
    (6) mapping_lr_mul: Learning rate multiplier for the mapping network.
        (default: 0.01)
    (7) repeat_w: Repeat w-code for different layers.

    Settings for the synthesis network:

    (1) resolution: The resolution of the output image.
    (2) image_channels: Number of channels of the output image. (default: 3)
    (3) final_tanh: Whether to use `tanh` to control the final pixel range.
        (default: False)
    (4) const_input: Whether to use a constant in the first convolutional layer.
        (default: True)
    (5) fused_scale: Whether to fused `upsample` and `conv2d` together,
        resulting in `conv2d_transpose`. (default: `auto`)
    (6) use_wscale: Whether to use weight scaling. (default: True)
    (7) noise_type: Type of noise added to the convolutional results at each
        layer. (default: `spatial`)
    (8) fmaps_base: Factor to control number of feature maps for each layer.
        (default: 16 << 10)
    (9) fmaps_max: Maximum number of feature maps in each layer. (default: 512)
    """
    def __init__(self,
                 # Settings for mapping network.
                 z_dim=256,
                 w_dim=256,
                 repeat_w=False,
                 normalize_z=False,
                 mapping_layers=3,
                 mapping_hidden_dim=256,
                 # Settings for conditional generation.
                 label_dim=0,
                 embedding_dim=512,
                 normalize_embedding=True,
                 normalize_embedding_latent=False,
                 label_concat=True,
                 # Settings for synthesis network.
                 resolution=-1,
                 synthesis_input_dim=3,
                 synthesis_output_dim=256,
                 synthesis_layers=8,
                 grid_scale=0.24,
                 feature_dim=3,
                 norm_feature=True,
                 # Setting for SuperResolution 
                 nerf_res=64,
                 bg_nerf_res=None,
                 use_sr=False,
                 noise_type='spatial',
                 fmaps_base=32 << 10,                                     
                 fmaps_max=512,
                 filter_kernel=(1, 3, 3, 1),
                 conv_clamp=None,
                 eps=1e-8,
                 # Setting for FG and BG
                 num_bbox=2,
                 background_only=False,
                 static_background=False,
                 background_path=None,
                 use_object=False,
                 use_hs=False,
                 fg_use_dirs=True,
                 ps_cfg=dict(),
                 hs_cfg=dict(),
                 vr_cfg=dict(),
                 ):
        """Initializes with basic settings.

        Raises:
            ValueError: If the `resolution` is not supported, or `fused_scale`
                is not supported.
        """
        super().__init__()

        self.z_dim = z_dim
        self.w_dim = w_dim
        self.repeat_w = repeat_w
        self.normalize_z = normalize_z
        self.mapping_layers = mapping_layers


        self.label_dim = label_dim
        self.embedding_dim = embedding_dim
        self.normalize_embedding = normalize_embedding
        self.normalize_embedding_latent = normalize_embedding_latent

        self.resolution = resolution
        self.nerf_res = nerf_res
        self.bg_nerf_res = bg_nerf_res or nerf_res
        self.use_bg_up = (self.bg_nerf_res != self.nerf_res)
        bg_up_layers = int(2*np.log2(self.nerf_res//self.bg_nerf_res))+1
        self.bg_synthesis_layers = synthesis_layers // 2
        self.num_layers = max(synthesis_layers, self.bg_synthesis_layers+bg_up_layers+1)
        self.eps = eps
        self.use_object = use_object
        self.use_hs = use_hs
        self.use_sr = use_sr

        if self.repeat_w:
            self.mapping_space_dim = self.w_dim
        else:
            self.mapping_space_dim = self.w_dim * (self.num_layers + 1)

        # Dimension of latent space, which is convenient for sampling.
        self.num_bbox = num_bbox
        self.latent_dim = (num_bbox+1, z_dim)

        self.background_only = background_only
        self.static_background = static_background
        if self.static_background:
            assert background_path is not None
            print('we use staic background!!')
            bk_image = load_image(background_path)[:,:,:3]
            # , (self.resolution, self.resolution)
            bk_image = torch.from_numpy(bk_image.astype(np.float32))
            bk_image = bk_image/255.0 
            self.bk_image = bk_image.unsqueeze(0)

        self.mapping = MappingNetwork(input_dim=z_dim,
                                      output_dim=w_dim,
                                      num_outputs=self.num_layers+1,
                                      repeat_output=repeat_w,
                                      normalize_input=normalize_z,
                                      num_layers=mapping_layers,
                                      hidden_dim=mapping_hidden_dim,
                                      label_dim=label_dim,
                                      embedding_dim=embedding_dim,
                                      normalize_embedding=normalize_embedding,
                                      normalize_embedding_latent=normalize_embedding_latent,
                                      eps=eps,
                                      label_concat=label_concat,
                                      lr=None)
        
        # This is used for truncation trick.
        if self.repeat_w:
            self.register_buffer('w_avg', torch.zeros(w_dim))
        else:
            self.register_buffer('w_avg', torch.zeros(self.num_layers * w_dim))

        self.pointsampler = PointsBboxSampling(**ps_cfg)
        self.hierachicalsampler = HierarchicalBboxSampling(**hs_cfg)
        self.volumerenderer = RendererBbox(**vr_cfg)

        if self.background_only:
            self.synthesis = SynthesisNetwork(w_dim=w_dim,
                                               in_channels=synthesis_input_dim,
                                               num_layers=self.bg_synthesis_layers,
                                               out_channels=synthesis_output_dim//2,
                                               grid_scale=grid_scale,
                                               feature_dim=feature_dim,
                                               norm_feature=norm_feature,
                                               use_dirs=False,
                                               eps=eps,
                                               )
        else:
            self.synthesis = SynthesisNetwork(w_dim=w_dim,
                                              in_channels=synthesis_input_dim,
                                              num_layers=synthesis_layers,
                                              out_channels=synthesis_output_dim,
                                              grid_scale=grid_scale,
                                              use_dirs=fg_use_dirs,
                                              feature_dim=feature_dim,
                                              eps=eps)
            self.bg_synthesis = SynthesisNetwork(w_dim=w_dim,
                                               in_channels=synthesis_input_dim,
                                               num_layers=self.bg_synthesis_layers,
                                               out_channels=synthesis_output_dim//2,
                                               grid_scale=grid_scale,
                                               feature_dim=feature_dim,
                                               # use_dirs=True,
                                               eps=eps)
        if self.use_bg_up:
            self.bg_superresolution = SuperResolution(in_channels=feature_dim,
                                                      w_dim=w_dim,
                                                      input_res=self.bg_nerf_res,
                                                      image_res=self.nerf_res,
                                                      fmaps_base=fmaps_base,
                                                      fmaps_max=fmaps_max,
                                                      noise_type=noise_type,
                                                      filter_kernel=filter_kernel,
                                                      out_channels=feature_dim,
                                                      conv_clamp=conv_clamp,
                                                      eps=eps)
        self.nerf_feature_dim = feature_dim
        if self.use_sr:
            assert feature_dim > 3, 'input dim of sr module must be larger than 3!'
            self.superresolution = SuperResolution(in_channels=feature_dim,
                                                   input_res=nerf_res,
                                                   image_res=resolution,
                                                   fmaps_base=fmaps_base,
                                                   fmaps_max=fmaps_max,
                                                   noise_type=noise_type,
                                                   filter_kernel=filter_kernel,
                                                   conv_clamp=conv_clamp,
                                                   eps=eps)
        self.init_weights()

    def init_weights(self):
        self.mapping.init_weights()
        self.synthesis.init_weights()
        if hasattr(self, 'bg_synthesis'):
            print('init_bg_synthesis!')
            self.bg_synthesis.init_weights()

    def forward(self,
                z,
                label=None,
                lod=None,
                w_moving_decay=None,
                sync_w_avg=False,
                style_mixing_prob=None,
                trunc_psi=None,
                trunc_layers=None,
                enable_amp=False,
                noise_std=0,
                foreground_only=False,
                background_only=False,
                noise_mode='const',
                fused_modulate=False,                                            
                impl='cuda', 
                fp16_res=None,
                ps_kwargs=dict(),
                hs_kwargs=dict(),
                vr_kwargs=dict(),
                bbox_kwargs=dict()):

        resolution = self.resolution
        lod = self.synthesis.lod.cpu().tolist() if lod is None else lod

        reshape_z = z.reshape(-1, self.latent_dim[-1])
        mapping_results = self.mapping(reshape_z, label)

        w = mapping_results['w']

        if self.training and w_moving_decay is not None:
            if sync_w_avg:
                batch_w_avg = all_gather(w.detach()).mean(dim=0)
            else:
                batch_w_avg = w.detach().mean(dim=0)
            self.w_avg.copy_(batch_w_avg.lerp(self.w_avg, w_moving_decay))

        wp = mapping_results.pop('wp')
         
        if not self.training:
            trunc_psi = 1.0 if trunc_psi is None else trunc_psi
            trunc_layers = 0 if trunc_layers is None else trunc_layers
            if trunc_psi < 1.0 and trunc_layers > 0:
                w_avg = self.w_avg.reshape(1, -1, self.w_dim)[:, :trunc_layers]
                wp[:, :trunc_layers] = w_avg.lerp(
                    wp[:, :trunc_layers], trunc_psi)

        def mask_forward(wp, dirs, pts, mask, lod):
            device = wp.device
            NS, C = pts.shape[-2:]
            broad_mask = mask[..., None, None].expand(tuple(pts.shape[:-1])+(1,))
            index = torch.nonzero(mask)

            import time
            torch.cuda.synchronize()
            start = time.time()

            final_rgb = torch.zeros(tuple(pts.shape[:-1])+(self.nerf_feature_dim,), device=pts.device, dtype=pts.dtype)
            final_sigma = torch.ones(tuple(pts.shape[:-1])+(1,), device=pts.device, dtype=pts.dtype) * -1e2
            # print(final_sigma.mean())
            node0 = time.time()
            # print('build tensor time', node0-start) 
            # final_rgb = final_rgb.to(pts.dtype).to(pts.device)
            # final_sigma = final_sigma.to(pts.dtype).to(pts.device)
            # torch.cuda.synchronize()
            node1 = time.time()
            # print('dtype tensor time', node1-node0) 
            mask_pts = torch.masked_select(pts,broad_mask).reshape(-1, NS, C)
            mask_wp = wp[index[:, 0], index[:, 1]]
            mask_dirs = dirs[index[:, 0], index[:,1], index[:,2], index[:,3]]
            # torch.cuda.synchronize()

            node2 = time.time()
            # print('select input time', node2-node1)
            mask_results = self.synthesis(wp=mask_wp,
                                               pts=mask_pts,
                                               dirs=mask_dirs,
                                               lod=lod)
            # torch.cuda.synchronize()
            node3 = time.time()
            # print('network forward time', node3-node2)

            final_sigma[broad_mask[...,0:1]] = mask_results['sigma'].reshape(-1)
            rgb_broad_mask = broad_mask.expand(tuple(pts.shape[:-1])+(self.nerf_feature_dim,))
            import time
            # torch.cuda.synchronize()
            start = time.time()
            final_rgb[rgb_broad_mask] = mask_results['rgb'].reshape(-1)
            import time
            # torch.cuda.synchronize()
            # print('select_rgb_time', time.time()-start)
            
            return final_rgb, final_sigma

        def bg_mask_forward(network, wp, dirs, pts, ratio, lod):
            mask = None
            if ratio != 1:
                device = wp.device
                NS, C = pts.shape[-2:]

                final_rgb = torch.zeros(tuple(pts.shape[:-1])+(self.nerf_feature_dim,), device=pts.device, dtype=pts.dtype)
                final_sigma = torch.ones(tuple(pts.shape[:-1])+(1,), device=pts.device, dtype=pts.dtype) * -1e2
                bs, H, W, _ = dirs.shape
                bH = int(ratio * H)
                bW = W
                mask_pts = pts[:,:,(bW-bH)//2:(bW+bH)//2]
                mask_dirs = dirs[:,(bW-bH)//2:(bW+bH)//2]
            else:
                mask_pts = pts
                mask_dirs = dirs
            mask_results = network(wp=wp,
                                   pts=mask_pts,
                                   dirs=mask_dirs,
                                   lod=lod)
            if ratio != 1:
                # final_sigma[broad_mask[...,0:1]] = mask_results['sigma'].reshape(-1)
                # rgb_broad_mask = broad_mask.expand(tuple(pts.shape[:-1])+(self.nerf_feature_dim,))
                # final_rgb[rgb_broad_mask] = mask_results['rgb'].reshape(-1)
                final_sigma[:,:,(bW-bH)//2:(bW+bH)//2] = mask_results['sigma']
                final_rgb[:,:,(bW-bH)//2:(bW+bH)//2] = mask_results['rgb']
            else:
                final_rgb = mask_results['rgb']
                final_sigma = mask_results['sigma']

            return final_rgb, final_sigma

        _, N, w_dim = wp.shape
        wp = wp.reshape(-1, self.num_bbox + 1, N, w_dim)
        fg_wp = wp[:, :self.num_bbox]
        bg_wp = wp[:, self.num_bbox]
        # ps_results = Points_Sampling() 
        # synthesis_results = self.synthesis(wp, pts, dirs, lod=None)
        # hierichical_sampling = self.HS(synthesis_results)
        # fine_synthesis_results = 
        # image = xxxxxx 
        with autocast(enabled=enable_amp):
            if not self.background_only:
                ps_results = self.pointsampler(batch_size=wp.shape[0],
                                               resolution=ps_kwargs.get('test_resolution', self.nerf_res),
                                               bg_resolution=self.bg_nerf_res,
                                               bbox_kwargs=bbox_kwargs,
                                               **ps_kwargs)
                if 'lock_view' in bbox_kwargs:
                    fg_dirs = ps_results['ray_dirs_lock']
                    # fg_dirs = torch.zeros_like(ps_results['ray_dirs_object'])
                    # fg_dirs[..., -1] = 1
                else:
                    fg_dirs = ps_results['ray_dirs_object']
                fg_coarse_rgbs, fg_coarse_sigmas = mask_forward(wp=fg_wp, 
                                                dirs=fg_dirs, 
                                                pts=ps_results['fg_pts_object'], 
                                                mask=ps_results['ray_mask'], 
                                                lod=lod)
                # background 
                # bg_synthesis_results = self.bg_synthesis(wp=bg_wp,
                #                                          dirs=ps_results['ray_dirs_world'],
                #                                          pts=ps_results['bg_pts'],
                #                                          lod=lod)
                # TODO there is a bug f**k
                bg_rgbs, bg_sigmas = bg_mask_forward(
                                                network=self.bg_synthesis,
                                                wp=bg_wp, 
                                                dirs=ps_results['ray_dirs_world'],
                                                pts=ps_results['bg_pts'], 
                                                # mask=ps_results.get('bg_ray_mask', None), 
                                                ratio = self.pointsampler.aspect_ratio,
                                                lod=lod)
                if self.use_hs:
                    # TODO bbox transoform it to bbox coordinate
                    fg_hs_results = self.hierachicalsampler(coarse_rgbs=fg_coarse_rgbs,
                                                        coarse_sigmas=fg_coarse_sigmas,
                                                        pts_z=ps_results['fg_pts_depth'],
                                                        ray_oris=ps_results['ray_oris_object'],
                                                        ray_dirs=ps_results['ray_dirs_object'],
                                                        noise_std=noise_std,
                                                        max_depth=0,
                                                        **hs_kwargs)
                    # test
                    fine_dirs = ps_results['ray_dirs_object'].to(ps_results['ray_dirs_object'].dtype)
                    fine_pts = fg_hs_results['pts'].to(ps_results['fg_pts_object'].dtype)
                    fg_fine_rgbs, fg_fine_sigmas = mask_forward(wp=fg_wp, 
                                                    dirs=fine_dirs, 
                                                    pts=fine_pts, 
                                                    mask=ps_results['ray_mask'], 
                                                    lod=lod)
                    
                    fg_rgbs = torch.cat((fg_fine_rgbs, fg_coarse_rgbs), dim=-2)
                    fg_sigmas = torch.cat((fg_fine_sigmas, fg_coarse_sigmas), dim=-2)
                    fg_pts_depth = torch.cat((fg_hs_results['pts_depth'], ps_results['fg_pts_depth']), dim=-2)
                    # TODO maybe we can remove it
                    _, indices = torch.sort(fg_pts_depth, dim=-2)
                    rgbs = torch.gather(fg_rgbs, -2, indices.expand(-1,-1,-1,-1,-1,3))
                    sigmas = torch.gather(fg_sigmas, -2, indices)
                    fg_pts_depth = torch.gather(fg_pts_depth, -2, indices)
                else:
                    fg_rgbs = fg_coarse_rgbs
                    fg_sigmas = fg_coarse_sigmas 
                    fg_pts_depth = ps_results['fg_pts_depth']
                # render foreground
                fg_render_results = self.volumerenderer(rgbs=fg_rgbs, 
                                                        sigmas=fg_sigmas, 
                                                        pts_z=fg_pts_depth, 
                                                        noise_std=noise_std,
                                                        max_depth=1e-3,
                                                        ray_mask=ps_results['ray_mask'],
                                                        **vr_kwargs)    
                avg_weights = fg_render_results['avg_weights']
                # render background
                bg_render_results = self.volumerenderer(rgbs=bg_rgbs,
                                                        sigmas=bg_sigmas,
                                                        pts_z=ps_results['bg_pts_depth'],
                                                        noise_std=noise_std,
                                                        max_depth=1e10,
                                                        # ray_mask=ps_results['ray_mask'],
                                                        ray_mask=None,
                                                        **vr_kwargs)
                # torch.cuda.synchronize()
                # print('volume render time:', vtime)
                if foreground_only:
                    print('fg_only!')
                    blend_rgb = fg_render_results['rgb']
                elif background_only:
                    print('bk_only!')
                    blend_rgb = bg_render_results['rgb']
                else:
                    if self.static_background:
                        fg_image = fg_render_results['rgb']
                        bk_image = self.bk_image.to(fg_image.dtype).to(fg_image.device)
                        if bk_image.shape[1:3] != fg_image.shape[1:3]:
                            bk_image = bk_image.permute(0, 3, 1, 2)
                            bk_image = F.interpolate(bk_image, fg_image.shape[1:3], mode='bilinear').permute(0, 2, 3, 1)
                        # print(fg_image.shape, fg_render_results['last_weights'].shape, bk_image.shape)
                        blend_rgb = fg_image + fg_render_results['last_weights'] * bk_image.detach()
                    else:
                        if self.use_bg_up:
                            bk_feature_image = bg_render_results['rgb']
                            bk_feature_image = bk_feature_image.permute(0, 3, 1, 2)
                            # bk_feature_image = 2*bk_feature_image - 1
                            bk_rgb_image = bk_feature_image[:, :3]
                            bk_image_results = self.bg_superresolution(x=bk_feature_image,
                                                                       rgb=bk_rgb_image,
                                                                       wp=bg_wp[:, self.bg_synthesis_layers+1:],
                                                                       lod=None,
                                                                       noise_mode=noise_mode,
                                                                       fused_modulate=fused_modulate,
                                                                       impl=impl,
                                                                       fp16_res=fp16_res)
                            # bk_image = bk_results['nerf_image'].contiguous()
                            # TODO consider feature scale? sigmoid???
                            bg_image = bk_image_results['image'].contiguous().permute(0, 2,3 ,1)
                            bg_up_feature = bk_image_results['feature'].contiguous().permute(0, 2, 3, 1)
                            bg_up_feature[...,:3] = bg_image 
                            # print('bk_act', bk_feature_image.mean().item(), bk_feature_image.std().item())
                            # print('bk_up', bg_up_feature.mean().item(), bg_up_feature.std().item())

                            # print('bk_up_sig', bg_up_feature.sigmoid().mean().item(), bg_up_feature.sigmoid().std().item())
                            # fg_se = torch.masked_select( fg_render_results['rgb'], fg_render_results['last_weights']!=1) 
                            # print('fg_act', fg_se.mean().item(), fg_se.std().item())
                            blend_rgb = fg_render_results['rgb']+ fg_render_results['last_weights'] * bg_up_feature
                        else:
                            # fg_se = torch.masked_select( fg_render_results['rgb'], fg_render_results['last_weights']!=1)
                            # print('bk_act', bg_render_results['rgb'].mean().item(), bg_render_results['rgb'].std().item())
                            # print('fg_act', fg_se.mean().item(), fg_se.std().item())

                            blend_rgb = fg_render_results['rgb'] + fg_render_results['last_weights'] * bg_render_results['rgb']
                # blend_rgb = torch.zeros_like(fg_render_results['rgb']) + bg_render_results['rgb']
            else:
                ps_results = self.pointsampler(batch_size=wp.shape[0],
                                               resolution=ps_kwargs.get('test_resolution', self.nerf_res),
                                               bg_resolution=self.bg_nerf_res,
                                               bbox_kwargs=bbox_kwargs,
                                               **ps_kwargs)
                # background 
                # bg_synthesis_results = self.synthesis(wp=bg_wp,
                #                                          dirs=ps_results['ray_dirs_world'],
                #                                          pts=ps_results['bg_pts'],
                #                                          lod=lod)
                bg_rgbs, bg_sigmas = bg_mask_forward(
                                                network=self.synthesis,
                                                wp=bg_wp, 
                                                dirs=ps_results['ray_dirs_world'],
                                                pts=ps_results['bg_pts'], 
                                                # mask=ps_results.get('bg_ray_mask', None), 
                                                ratio = self.pointsampler.aspect_ratio,
                                                lod=lod)
                # render background
                bg_render_results = self.volumerenderer(rgbs=bg_rgbs,
                                                        sigmas=bg_sigmas,
                                                        pts_z=ps_results['bg_pts_depth'],
                                                        noise_std=noise_std,
                                                        max_depth=1e10,
                                                        **vr_kwargs)
                if self.use_bg_up:
                    bk_feature_image = bg_render_results['rgb']
                    bk_feature_image = bk_feature_image.permute(0, 3, 1, 2)
                    # bk_feature_image = 2*bk_feature_image - 1
                    bk_rgb_image = bk_feature_image[:, :3]
                    bk_image_results = self.bg_superresolution(x=bk_feature_image,
                                                               rgb=bk_rgb_image,
                                                               wp=bg_wp[:, self.bg_synthesis_layers+1:],
                                                               lod=None,
                                                               noise_mode=noise_mode,
                                                               fused_modulate=fused_modulate,
                                                               impl=impl,
                                                               fp16_res=fp16_res)
                    # bk_image = bk_results['nerf_image'].contiguous()
                    # TODO consider feature scale? sigmoid???
                    bg_image = bk_image_results['image'].contiguous().permute(0, 2,3 ,1)
                    bg_up_feature = bk_image_results['feature'].contiguous().permute(0, 2, 3, 1)
                    bg_up_feature[...,:3] = bg_image
                    blend_rgb = bg_up_feature
                else:
                    blend_rgb = bg_render_results['rgb']
                avg_weights = torch.zeros((1,), dtype=blend_rgb.dtype)
        # output
        # TODO Check value of feature range
        feature_image = blend_rgb.permute(0, 3, 1, 2)
        # TODO check it???
        # feature_image = 3*feature_image - 1
        if self.use_sr:
            rgb_image = feature_image[:, :3]
            image_results = self.superresolution(x=feature_image,
                                         rgb=rgb_image,
                                         wp=None,
                                         lod=lod,
                                         noise_mode=noise_mode,
                                         fused_modulate=fused_modulate,
                                         impl=impl,
                                         fp16_res=fp16_res)
            rgb_image = image_results['nerf_image'].contiguous()
            image = image_results['image'].contiguous()
        else:
            rgb_image = feature_image
            image = feature_image

        camera = torch.cat([ps_results['pitch'], ps_results['yaw']], -1)
        results = {**mapping_results, 'image': image, 'image_raw': rgb_image, 'latent': z, 'camera': camera, 'avg_weights': avg_weights.reshape(-1, 1), 'ray_mask': ps_results['ray_mask']}
        if self.use_object:
            results.update(image_bbox=bbox_kwargs['g_image_bbox'])
        return results

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
                 label_dim,
                 embedding_dim,
                 normalize_embedding,
                 normalize_embedding_latent,
                 eps,
                 label_concat,
                 lr=None):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_outputs = num_outputs
        self.repeat_output = repeat_output
        self.normalize_input = normalize_input
        self.num_layers = num_layers
        # self.out_channels = out_channels
        # TODO 
        # self.lr_mul = lr_mul

        self.label_dim = label_dim
        self.embedding_dim = embedding_dim
        self.normalize_embedding = normalize_embedding
        self.normalize_embedding_latent = normalize_embedding_latent
        self.eps = eps
        self.label_concat = label_concat

        self.norm = PixelNormLayer(dim=-1, eps=eps)

        if num_outputs is not None and not repeat_output:
            output_dim = output_dim * num_outputs

        if self.label_dim > 0:
            if self.label_concat:
                input_dim = input_dim + embedding_dim
                self.embedding = EqualLinear(label_dim, 
                                            embedding_dim, 
                                            bias=True, 
                                            bias_init=0, 
                                            lr_mul=1)
            else:
                self.embedding = EqualLinear(label_dim, 
                                            output_dim, 
                                            bias=True, 
                                            bias_init=0, 
                                            lr_mul=1)

        
        
        network = []
        
        for i in range(num_layers):
            in_channels = (input_dim if i == 0 else hidden_dim)
            out_channels = (output_dim if i == (num_layers - 1) else hidden_dim)
            network.append(nn.Linear(in_channels, out_channels))
            network.append(nn.LeakyReLU(0.2, inplace=True))
        self.network = nn.Sequential(*network)
    
    def init_weights(self):
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight,
                                        a=0.2,
                                        mode='fan_in',
                                        nonlinearity='leaky_relu')

    def forward(self, z, label=None):
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
            
            embedding = self.embedding(label)
            if self.normalize_embedding and self.label_concat:
                embedding = self.norm(embedding)
            if self.label_concat:
                w = torch.cat((z, embedding), dim=1)
            else:
                w = z
        else:
            w = z

        if self.label_dim > 0 and self.normalize_embedding_latent and self.label_concat:
            w = self.norm(w)

        for layer in self.network:
            w = layer(w)
        
        if self.label_dim > 0 and (not self.label_concat):
            w = w * embedding

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
    """Same architecture as TALLSIREN but adds a UniformBoxWarp to map input points to -1, 1"""
    def __init__(self,
                 w_dim,
                 in_channels,
                 num_layers,
                 out_channels,
                 grid_scale=0.24,
                 use_dirs=True,
                 feature_dim=3,
                 norm_feature=True,
                 eps=1e-8):
        super().__init__()

        self.in_channels = in_channels
        self.w_dim = w_dim
        self.out_channels = out_channels  
        self.use_dirs = use_dirs
        self.feature_dim = feature_dim
        self.norm_feature = norm_feature
 
        self.register_buffer('lod', torch.zeros(()))

        self.gridwarper = UniformBoxWarp(
            grid_scale 
        )  # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.

        network = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            out_channels = out_channels
            film = FiLMLayer(in_channels, out_channels, w_dim)
            network.append(film)
        self.mlp_network = nn.Sequential(*network)
        self.sigma_head = nn.Linear(out_channels, 1)
        if self.use_dirs:
            self.color_film = FiLMLayer(out_channels+3, out_channels, w_dim)
        else:
            self.color_film = FiLMLayer(out_channels, out_channels, w_dim)
        self.color_head = nn.Linear(out_channels, feature_dim)


    def init_weights(self,):
        self.sigma_head.apply(freq_init(25))
        self.color_head.apply(freq_init(25))

        for module in self.modules():
            if isinstance(module, FiLMLayer):
                module.init_weights()
        
        self.mlp_network[0].init_weights(first=True)


    def forward(self, wp, pts, dirs, lod=None):
        num_dims = pts.ndim
        assert num_dims in [3, 4, 5, 6]
        if num_dims == 5:
             bs, H, W, D, C = pts.shape 
             dirs = dirs.unsqueeze(-2).expand_as(pts)
             pts = pts.reshape(bs, H*W*D, C)
             dirs = dirs.reshape(bs, H*W*D, C)
        elif num_dims == 4:
            bs, N, D, C = pts.shape 
            dirs = dirs.unsqueeze(-2).expand_as(pts)
            pts = pts.reshape(bs, N*D, C)
            dirs = dirs.reshape(bs, N*D, C)
        elif num_dims == 3:
            dirs = dirs.unsqueeze(-2).expand_as(pts)
        elif num_dims == 6:
             bs, _, H, W, D, C = pts.shape
             dirs = dirs.unsqueeze(1).unsqueeze(-2).expand_as(pts)
             pts = pts.reshape(bs, H*W*D, C)
             dirs = dirs.reshape(bs, H*W*D, C)

        x = self.gridwarper(pts)

        for idx, layer in enumerate(self.mlp_network):
            x = layer(x, wp[:, idx])
           
        sigma = self.sigma_head(x)
        if self.use_dirs:
            dirs = torch.cat([x, dirs], dim=-1)
        else:
            dirs = x
        color = self.color_film(dirs, wp[:, len(self.mlp_network)])

        import time
        # torch.cuda.synchronize()
        start = time.time()
        # TODO switch off sigmoid??? I am not sure about it
        color = self.color_head(color).sigmoid()
        color = 2*color - 1
        # color[...,:3] = color[...,:3].sigmoid()
        # if self.norm_feature:
        #     color[...,3:] = color[...,3:].sigmoid()
        # print(color.mean().item())
        # TODO debug
        # color = color.sigmoid()
        import time
        # torch.cuda.synchronize()
        # print('color_head time', time.time()-start)
        
        if num_dims == 5:
            sigma = sigma.reshape(bs, H, W, D, sigma.shape[-1])
            color = color.reshape(bs, H, W, D, color.shape[-1])
        elif num_dims == 4:
            sigma = sigma.reshape(bs, N, D, sigma.shape[-1])
            color = color.reshape(bs, N, D, color.shape[-1])
        elif num_dims == 6:
            sigma = sigma.reshape(bs, 1, H, W, D, sigma.shape[-1])
            color = color.reshape(bs, 1, H, W, D, color.shape[-1])
        results = {
            'sigma': sigma,
            'rgb': color,
        }
        return results

class SuperResolution(nn.Module):
    def __init__(self, 
                 in_channels,
                 input_res,
                 image_res,
                 fmaps_max,
                 fmaps_base,
                 noise_type,
                 filter_kernel,
                 conv_clamp,
                 eps,
                 w_dim=0,
                 out_channels=None,
                 kernel_size=3,
                 ):
        super().__init__()
        self.input_res = input_res
        self.image_res = image_res
        self.noise_type = noise_type
        self.filter_kernel = filter_kernel
        self.fmaps_max = fmaps_max
        self.fmaps_base = fmaps_base
        
        self.num_stages = int(np.log2(image_res//input_res))

        self.register_buffer('lod', torch.zeros(()))

        for lod_idx in range(self.num_stages-1, -1, -1):
            block_idx = self.num_stages - lod_idx - 1
            cur_res = input_res * (2**block_idx)
            in_chs = self.get_nf(cur_res//2) if block_idx!=0 else in_channels
            out_chs = self.get_nf(cur_res)
           
            scale_factor = 2
            block_res = cur_res * scale_factor
            if lod_idx == 0:
                final_dim = out_channels
            else:
                final_dim = None
            block = SynthesisBlock(in_chs, 
                                   out_chs, 
                                   w_dim=w_dim, 
                                   resolution=block_res,
                                   kernel_size=kernel_size,
                                   scale_factor=scale_factor,
                                   filter_kernel=filter_kernel,
                                   noise_type=noise_type,
                                   activation_type='lrelu',
                                   conv_clamp=conv_clamp,
                                   eps=eps,
                                   final_dim=final_dim,
                                   )
            layer_name = f'block{block_idx}'
            self.add_module(layer_name, block)
   
        self.register_buffer('lod', torch.zeros(()))
        # Used for upsampling output images for each resolution block for sum.
        self.register_buffer(                               
           'filter', upfirdn2d.setup_filter(filter_kernel))

    def get_nf(self, res):
        return min(self.fmaps_base // res, self.fmaps_max)

    def forward(self,
                x,
                rgb,
                wp,
                lod=None,
                noise_mode='const',
                fused_modulate=False,
                impl='cuda',
                fp16_res=None):

        lod = self.lod.item() if lod is None else lod
        assert lod <= self.num_stages 

        # if fp16_res is not None and self.input_res >= fp16_res:
        #     x = x.to(torch.float16)

        nerf_rgb, cur_rgb = rgb, rgb
        for cur_lod in range(self.num_stages, -1, -1):
            block_idx = self.num_stages - cur_lod - 1
            
            if lod<cur_lod+1: 
                if block_idx == -1 :
                    x, cur_rgb = x, cur_rgb
                    # print(f'forward_block, cur_rgb.shape:{cur_rgb.shape}')
                else:
                    block = getattr(self, f'block{block_idx}')
                    # print(f'block_idx/num_stages:{block_idx}/{self.num_stages},cur_rgb.type:{cur_rgb.dtype}')
                    # print(f'cur_lod:{cur_lod}', x.mean().item())
                    # print(f'cur_lod:{cur_lod}', cur_rgb.mean().item())
                    x, cur_rgb = block(x,
                                       cur_rgb,
                                       wp[:, 2*block_idx:2*block_idx+3] if wp is not None else None,
                                       fp16_res=fp16_res,
                                       noise_mode=noise_mode,
                                       fused_modulate=fused_modulate,
                                       impl=impl)
            # TODO study difference???
            up_mode = 'nearest'
            if up_mode == 'bilinear':
                up_kwargs = {'mode': 'bilinear',
                             'align_corners': False,
                             'antialias': True}
            elif up_mode == 'nearest':
                up_kwargs = {'mode': 'nearest'}
            else:
                raise NotImplementedError

            if cur_lod-1 < lod <= cur_lod:
                # print(f'block_idx:{block_idx}, forward, cur_rgb.shape:{cur_rgb.shape}')
                rgb = cur_rgb
            elif cur_lod < lod < cur_lod + 1:
                alpha = np.ceil(lod) - lod
                # TODO bilinear?
                rgb = F.interpolate(rgb, scale_factor=2, **up_kwargs) 
                rgb = cur_rgb * alpha + rgb * (1 - alpha)
                # print(f'block_idx:{block_idx}, blend, cur_rgb.shape:{cur_rgb.shape}')
            elif lod >= cur_lod + 1:
                # print(f'block_idx:{block_idx}, upsample, cur_rgb.shape:{rgb.shape}')
                rgb = F.interpolate(rgb, scale_factor=2, **up_kwargs)

        results = {'image': rgb,
                   'nerf_image': nerf_rgb,
                   'feature': x}
        return results

class SynthesisBlock(nn.Module):
    def __init__(self,
                 in_chs, 
                 out_chs, 
                 w_dim, 
                 resolution,
                 kernel_size,
                 scale_factor,
                 filter_kernel,
                 noise_type,
                 activation_type,
                 conv_clamp,
                 eps,
                 final_dim=None):
        super().__init__()
        self.w_dim = w_dim
        self.resolution = resolution
        self.scale_factor = scale_factor
        # Used for upsampling output images for each resolution block for sum.
        self.register_buffer(                                                          
            'filter', upfirdn2d.setup_filter(filter_kernel))

        # TODO demodulate is set False for remove adain
        layer_kwargs = {'demodulate': True,
                        'use_wscale': True,
                        'wscale_gain': 1.0,
                        'lr_mul': 1.0,
                        'activation_type': 'lrelu',
                        }
        self.conv0 = ModulateConvLayer(in_channels=in_chs,
                                         out_channels=out_chs,
                                         resolution=resolution,
                                         w_dim=w_dim,
                                         kernel_size=kernel_size,
                                         add_bias=True,
                                         scale_factor=scale_factor,
                                         filter_kernel=filter_kernel,
                                         noise_type=noise_type,
                                         conv_clamp=conv_clamp,
                                         eps=eps,
                                         **layer_kwargs,
                                         )
        self.conv1 = ModulateConvLayer(in_channels=out_chs,
                                         out_channels= final_dim or out_chs,
                                         resolution=resolution,
                                         w_dim=w_dim,
                                         kernel_size=kernel_size,
                                         add_bias=True,
                                         scale_factor=1,
                                         filter_kernel=None,
                                         noise_type=noise_type,
                                         conv_clamp=conv_clamp,
                                         eps=eps,
                                         **layer_kwargs,
                                         )
        rgb_layer_kwargs = {'demodulate': True,
                            'use_wscale': True,
                            'wscale_gain': 1.0 ,
                            'lr_mul': 1.0,
                            'activation_type': 'linear',
                            }
        self.torgb = ModulateConvLayer(in_channels=final_dim or out_chs,
                                         out_channels=3,
                                         resolution=resolution,
                                         w_dim=w_dim,
                                         kernel_size=1,
                                         add_bias=True,
                                         scale_factor=1,
                                         filter_kernel=None,
                                         noise_type='none',
                                         conv_clamp=conv_clamp,
                                         eps=eps,
                                         **rgb_layer_kwargs,
                                         )
    def forward(self,
                x,
                rgb,
                wp,
                fp16_res,
                noise_mode,
                fused_modulate,
                impl):
        assert (x.shape[-2] == x.shape[-1])
        input_res = x.shape[-2]

        dtype = torch.float32
        if fp16_res is not None and input_res >= fp16_res:
            # print(f'input_res:{input_res}', 'cast2fp16')
            dtype = torch.float16
            x = x.to(torch.float16)
        x = self.conv0(x, 
                       wp[:, 0] if self.w_dim!=0 else None , 
                       noise_mode=noise_mode,
                       fused_modulate=fused_modulate,
                       impl=impl)
        if isinstance(x, tuple): x = x[0]
        x = self.conv1(x, 
                       wp[:, 1] if self.w_dim!=0 else None, 
                       noise_mode=noise_mode,
                       fused_modulate=fused_modulate,
                       impl=impl)
        if isinstance(x, tuple): x = x[0]
        y = self.torgb(x,
                       wp[:, 2] if self.w_dim!=0 else None,
                       fused_modulate=fused_modulate,
                       impl=impl)
        if isinstance(y, tuple): y = y[0]
        y = y.to(torch.float32)
        if self.scale_factor > 1 and rgb is not None:
            rgb = y + upfirdn2d.upsample2d(rgb, self.filter, impl=impl)
        else:
            rgb = y + rgb if rgb is not None else y

        assert x.dtype == dtype
        assert rgb is None or rgb.dtype == torch.float32
        return x, rgb


class FiLMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, w_dim, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w_dim = w_dim 
        
        self.layer = nn.Linear(input_dim, output_dim)
        self.style = nn.Linear(w_dim, output_dim*2)

    def init_weights(self, first=False):
        # initial with 25 frequency
        if not first:
            self.layer.apply(freq_init(25))
        else:
            self.layer.apply(first_film_init)
        # kaiming initial && scale 1/4
        nn.init.kaiming_normal_(self.style.weight,
                                a=0.2,
                                mode='fan_in',
                                nonlinearity='leaky_relu')
        with torch.no_grad(): self.style.weight *= 0.25

    def extra_repr(self):
        return (f'in_ch={self.input_dim}, '
                f'out_ch={self.output_dim}, '
                f'w_ch={self.w_dim}')

    def forward(self, x, wp):
        x = self.layer(x)
        style = self.style(wp)
        style_split = style.unsqueeze(1).chunk(2, dim=2)
        freq = style_split[0]
        # Scale for sin activation
        freq = freq*15 + 30
        phase_shift = style_split[1]
        return torch.sin(freq * x + phase_shift)

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

class UniformBoxWarp(nn.Module):
    def __init__(self, sidelength, ):
        super().__init__()
        self.min_pts, self.max_pts = None, None
        if isinstance(sidelength, (list, tuple)):
             self.min_pts = torch.tensor(sidelength[0]) 
             self.max_pts = torch.tensor(sidelength[1]) 
        else:
            self.scale_factor = 2 / sidelength

    def forward(self, coordinates):
        if self.min_pts is None:
            return coordinates * self.scale_factor
        else:
            min_pts = self.min_pts.to(coordinates.dtype).to(coordinates.device)
            max_pts = self.max_pts.to(coordinates.dtype).to(coordinates.device)
            shape = (1,)*(len(coordinates.shape)-1) + (3,)
            min_pts = min_pts.reshape(shape)
            max_pts = max_pts.reshape(shape)
            normalize_pts = (coordinates-min_pts) / (max_pts - min_pts+1e-8)
            normalize_pts = 2 * normalize_pts - 1
            return normalize_pts

def first_film_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1/num_input, 1/num_input)

def freq_init(freq):
    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6/num_input)/freq, 
                                  np.sqrt(6/num_input)/freq)
    return init

class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1,
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )
        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )
