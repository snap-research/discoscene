# python3.7
"""Contains the implementation of generator described in DiscoScene.

Paper: https://arxiv.org/pdf/2212.11984.pdf

"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rendering import PointsBboxSampling, HierarchicalBboxSampling, RendererBbox
from third_party.stylegan2_official_ops import fma
from third_party.stylegan2_official_ops import bias_act
from third_party.stylegan2_official_ops import upfirdn2d
from third_party.stylegan2_official_ops import conv2d_gradfix
from .utils.ops import all_gather
from .stylegan2_generator import ModulateConvLayer, DenseLayer, StyleGAN2Generator

__all__ = ['DiscoSceneGenerator']

# Resolutions allowed.
_RESOLUTIONS_ALLOWED = [8, 16, 32, 64, 128, 256, 512, 1024]

# Architectures allowed.
_ARCHITECTURES_ALLOWED = ['resnet', 'skip', 'origin']

# pylint: disable=missing-function-docstring

class DiscoSceneGenerator(nn.Module):
    """
    Defines the generator network in DiscoScene.

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
                 # 
                 label_dim=0,
                 embedding_dim=512,
                 embedding_bias=True,
                 embedding_use_wscale=True,
                 embedding_wscale_gian=1.0,
                 embedding_lr_mul=1.0,
                 normalize_embedding=True,
                 normalize_embedding_latent=False,
                 # Settings for conditional generation.
                 resolution=-1,
                 synthesis_input_dim=3,
                 synthesis_output_dim=256,
                 synthesis_layers=8,
                 grid_scale=0.24,
                 feature_dim=3,
                 # Setting for SuperResolution
                 nerf_res=64,
                 bg_nerf_res=None,
                 use_sr=False,
                 noise_type='spatial',
                 fmaps_base=32 << 10,
                 fmaps_max=512,
                 filter_kernel=(1, 3, 3, 1),
                 conv_clamp=None,
                 embed_cfg=dict(input_dim=3,
                                max_freq_log2=10-1,
                                N_freqs=10),
                 eps=1e-8,
                 # Setting for FG and BG
                 num_bbox=2,
                 background_path=None,
                 norm_feature=False,
                 use_object=False,
                 use_hs=False,
                 fg_use_dirs=True,
                 bg_use_dirs=True,
                 ps_cfg=dict(),
                 hs_cfg=dict(),
                 vr_cfg=dict(),
                 # Unusable parameters
                 background_only=False,
                 static_background=False,
                 # sample bg
                 condition_sample=False,
                 bg_condition_bbox=False, 
                 condition_bbox_embed_cfg=dict(input_dim=3,
                                max_freq_log2=10-1,
                                N_freqs=10),
                 kernel_size=3,
                 use_mask=False,
                 fg_condition_bbox=False,
                 use_bbox_label=False,
                 use_triplane=False,
                 triplane_cfg=None,
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

        self.label_dim = label_dim
        self.embedding_dim = embedding_dim
        self.embedding_bias = embedding_bias
        self.embedding_use_wscale = embedding_use_wscale
        self.embedding_wscale_gain = embedding_wscale_gian
        self.embedding_lr_mul = embedding_lr_mul
        self.normalize_embedding = normalize_embedding
        self.normalize_embedding_latent = normalize_embedding_latent

        self.resolution = resolution
        self.nerf_res = nerf_res
        self.bg_nerf_res = bg_nerf_res or nerf_res
        self.use_bg_up = (self.bg_nerf_res != self.nerf_res)
        bg_up_layers = int(2*np.log2(self.nerf_res//self.bg_nerf_res))+1
        self.bg_synthesis_layers = synthesis_layers // 2
        self.use_triplane = use_triplane
        if self.use_triplane: 
            synthesis_layers = int(np.log2(triplane_cfg['resolution']//triplane_cfg['init_res']*2)*2)
        self.num_layers = max(synthesis_layers, self.bg_synthesis_layers+bg_up_layers+1)
        self.eps = eps
        self.use_object = use_object
        self.use_hs = use_hs
        self.use_sr = use_sr
        self.background_only = background_only
        self.static_background = static_background

        self.condition_sample = condition_sample
        self.bg_condition_bbox = bg_condition_bbox
        self.fg_condition_bbox = fg_condition_bbox
        self.use_mask = use_mask
        self.use_bbox_label = use_bbox_label

        self.mlp_type = 'modulatemlp'
        if self.use_triplane:
            self.mlp_type = 'mlp' 
            self.triplane_cfg = triplane_cfg 
        else:
            self.triplane_cfg = None 


        if self.repeat_w:
            self.mapping_space_dim = self.w_dim
        else:
            self.mapping_space_dim = self.w_dim * (self.num_layers + 1)

        # Dimension of latent space, which is convenient for sampling.
        self.num_bbox = num_bbox
        self.latent_dim = (num_bbox+1, z_dim)

        fg_input_dim = z_dim
        if self.condition_sample:
            fg_input_dim += z_dim
        if self.fg_condition_bbox:
            self.bbox_embedder = Embedder(**condition_bbox_embed_cfg)
            fg_input_dim += self.bbox_embedder.out_dim*2

        self.mapping = MappingNetwork(
            input_dim=fg_input_dim,
            output_dim=w_dim,
            num_outputs=self.num_layers+1,
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

        if self.condition_sample or self.fg_condition_bbox or self.use_bbox_label:
            if self.bg_condition_bbox:
                self.bbox_embedder = Embedder(**condition_bbox_embed_cfg) 
            
            self.bg_mapping = MappingNetwork(
                input_dim=z_dim+(self.bbox_embedder.out_dim*self.num_bbox*2) if self.bg_condition_bbox else z_dim,
                output_dim=w_dim,
                num_outputs=self.num_layers+1,
                repeat_output=repeat_w,
                normalize_input=normalize_z,
                num_layers=mapping_layers,
                hidden_dim=mapping_fmaps,
                use_wscale=mapping_use_wscale,
                wscale_gain=mapping_wscale_gain,
                lr_mul=mapping_lr_mul,
                label_dim=0,
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

        # TODO rewrite???
        self.pointsampler = PointsBboxSampling(**ps_cfg)
        self.hierachicalsampler = HierarchicalBboxSampling(**hs_cfg)
        self.volumerenderer = RendererBbox(**vr_cfg)

        s_input_dim = synthesis_input_dim
        h_dim = None
        if self.use_triplane:
            s_input_dim = self.triplane_cfg['image_channels'] // 3
            h_dim = 64
        self.synthesis = SynthesisNetwork(w_dim=w_dim,
                                          in_channels=s_input_dim,
                                          num_layers=synthesis_layers,
                                          out_channels=synthesis_output_dim,
                                          hidden_channels=h_dim,
                                          use_dirs=fg_use_dirs,
                                          feature_dim=feature_dim,
                                          embed_cfg=embed_cfg,
                                          eps=eps,
                                          mlp_type=self.mlp_type,
                                          triplane_cfg=self.triplane_cfg)
        self.bg_synthesis = SynthesisNetwork(w_dim=w_dim,
                                           in_channels=synthesis_input_dim,
                                           num_layers=self.bg_synthesis_layers,
                                           out_channels=synthesis_output_dim//2,
                                           feature_dim=feature_dim,
                                           embed_cfg=embed_cfg,
                                           use_dirs=bg_use_dirs,
                                           eps=eps)
        if self.use_bg_up:
            self.bg_superresolution = SuperResolution(in_channels=feature_dim,
                                                      w_dim=w_dim,
                                                      input_res=self.bg_nerf_res,
                                                      image_res=self.nerf_res,
                                                      fmaps_base=fmaps_base,
                                                      fmaps_max=fmaps_max,
                                                      noise_type='spatial',
                                                      filter_kernel=filter_kernel,
                                                      out_channels=feature_dim,
                                                      conv_clamp=conv_clamp,
                                                      eps=eps,)

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
                                                   kernel_size=kernel_size,
                                                   eps=eps)
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
                bbox_kwargs=dict(),
                enable_amp=False,
                ):
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
        resolution = self.resolution
        bs = z.shape[0]
        lod = self.synthesis.lod.cpu().tolist() if lod is None else lod
        if not self.condition_sample and not self.fg_condition_bbox and not self.use_bbox_label:
            reshape_z = z.reshape(-1, self.latent_dim[-1])
            mapping_results = self.mapping(reshape_z, label, impl=impl)
        else:
            fg_z = z[:,:-1]
            bg_z = z[:,-1:]
            if self.condition_sample:
                fg_z = torch.cat([fg_z, bg_z.repeat(1, self.num_bbox, 1)], dim=-1)
                fg_z = fg_z.reshape(-1, self.latent_dim[-1]*2)
            if self.fg_condition_bbox:
                bbox_centers = bbox_kwargs['g_bbox'].reshape(z.shape[0], self.num_bbox, 8, 3).mean(dim=-2)
                bbox_scales = bbox_kwargs['g_bbox_scale'].reshape(z.shape[0], self.num_bbox, -1)*2
                bbox_mask = ((bbox_kwargs['g_bbox_valid']+1)/2)[..., None]
                bbox_centers = bbox_mask * bbox_centers
                bbox_scales = bbox_mask * bbox_scales
                if 'bbox_c' in bbox_kwargs:
                    bbox_centers = bbox_kwargs['bbox_c']
                if 'bbox_s' in bbox_kwargs:
                    bbox_scales = bbox_kwargs['bbox_s']
                bbox_centers_embed = self.bbox_embedder(bbox_centers).to(fg_z.dtype)
                bbox_scales_embed = self.bbox_embedder(bbox_scales).to(fg_z.dtype)
                fg_z = torch.cat([fg_z, bbox_centers_embed, bbox_scales_embed], dim=-1)
            


            fg_z = fg_z.reshape(-1, fg_z.shape[-1])
            fg_label = bbox_kwargs.get('g_bbox_label', None)
            if fg_label is not None:
                fg_label = fg_label.reshape(-1, fg_label.shape[-1])
            mapping_results = self.mapping(fg_z, fg_label, impl=impl)

            # bg_z = z[:,-1:].reshape(-1, self.latent_dim[-1])
            bg_z = bg_z.reshape(-1, self.latent_dim[-1])
            if self.bg_condition_bbox:
                # TODO normalize????
                bbox_centers = bbox_kwargs['g_bbox'].reshape(z.shape[0], self.num_bbox, 8, 3).mean(dim=-2)
                bbox_scales = bbox_kwargs['g_bbox_scale'].reshape(z.shape[0], self.num_bbox, -1)*2
                bbox_mask = ((bbox_kwargs['g_bbox_valid']+1)/2)[..., None]
                bbox_centers = bbox_mask * bbox_centers
                bbox_scales = bbox_mask * bbox_scales


                bbox_centers_embed = self.bbox_embedder(bbox_centers).reshape(bbox_centers.shape[0], -1).to(bg_z.dtype)
                bbox_scales_embed = self.bbox_embedder(bbox_scales).reshape(bbox_scales.shape[0], -1).to(bg_z.dtype)
                bg_z = torch.cat([bg_z, bbox_centers_embed, bbox_scales_embed], dim=-1)
            bg_mapping_results = self.bg_mapping(bg_z, label, impl=impl)
          

        w = mapping_results['w']
        if self.condition_sample or self.fg_condition_bbox or self.use_bbox_label:
            bg_w = mapping_results['w']
            w = torch.cat([w, bg_w], dim=0) 

        if self.training and w_moving_decay is not None:
            if sync_w_avg:
                batch_w_avg = all_gather(w.detach()).mean(dim=0)
            else:
                batch_w_avg = w.detach().mean(dim=0)
            self.w_avg.copy_(batch_w_avg.lerp(self.w_avg, w_moving_decay))

        #TODO revise stylemixing and trunc
        wp = mapping_results.pop('wp')
        if self.condition_sample or self.fg_condition_bbox or self.use_bbox_label:
            bg_wp = bg_mapping_results['wp']
            _fg_wp = wp
            _bg_wp = bg_wp
            wp = torch.cat([wp, bg_wp], dim=0) 


        if not self.training:
            trunc_psi = 1.0 if trunc_psi is None else trunc_psi
            trunc_layers = 0 if trunc_layers is None else trunc_layers
            if trunc_psi < 1.0 and trunc_layers > 0:
                w_avg = self.w_avg.reshape(1, -1, self.w_dim)[:, :trunc_layers]
                wp[:, :trunc_layers] = w_avg.lerp(
                    wp[:, :trunc_layers], trunc_psi)

        def fg_mask_forward(wp, dirs, pts, mask, lod):
            device = wp.device
            NS, C = pts.shape[-2:]
            broad_mask = mask[..., None, None].expand(tuple(pts.shape[:-1]) + (1,))
            index = torch.nonzero(mask)

            final_rgb = torch.zeros(tuple(pts.shape[:-1]) + (self.nerf_feature_dim,), device=pts.device,
                                    dtype=pts.dtype)
            final_sigma = torch.ones(tuple(pts.shape[:-1]) + (1,), device=pts.device, dtype=pts.dtype) * -1e2
            if mask.sum()>0:
                mask_pts = torch.masked_select(pts, broad_mask).reshape(-1, NS, C)
                # TODO
                # mask_wp = wp[index[:, 0], index[:, 1]]
                mask_wp = wp
                mask_dirs = dirs[index[:, 0], index[:, 1], index[:, 2], index[:, 3]]

                mask_results = self.synthesis(wp=mask_wp,
                                              pts=mask_pts,
                                              dirs=mask_dirs,
                                              lod=lod,
                                              mask=mask,
                                              box_warp=self.pointsampler.voxel_size*1.1)

                final_sigma[broad_mask[..., 0:1]] = mask_results['sigma'].reshape(-1)
                rgb_broad_mask = broad_mask.expand(tuple(pts.shape[:-1]) + (self.nerf_feature_dim,))
                final_rgb[rgb_broad_mask] = mask_results['rgb'].reshape(-1)

            return final_rgb, final_sigma

        def bg_forward(network, wp, dirs, pts, ratio, lod):
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
                final_sigma[:,:,(bW-bH)//2:(bW+bH)//2] = mask_results['sigma']
                final_rgb[:,:,(bW-bH)//2:(bW+bH)//2] = mask_results['rgb']
            else:
                final_rgb = mask_results['rgb']
                final_sigma = mask_results['sigma']

            return final_rgb, final_sigma

        _, N, w_dim = wp.shape
        if not self.condition_sample and not self.fg_condition_bbox or not self.use_bbox_label:
            wp = wp.reshape(-1, self.num_bbox + 1, N, w_dim)
            fg_wp = wp[:, :self.num_bbox]
            bg_wp = wp[:, self.num_bbox]
        else:
            fg_wp = wp[:self.num_bbox*bs].reshape(-1, self.num_bbox, N, w_dim)
            bg_wp = wp[self.num_bbox*bs:].reshape(-1, N, w_dim)

        ps_results = self.pointsampler(batch_size=fg_wp.shape[0],
                                       resolution=ps_kwargs.get('test_resolution', self.nerf_res),
                                       bg_resolution=self.bg_nerf_res,
                                       bbox_kwargs=bbox_kwargs,
                                       **ps_kwargs)
        if 'lock_view' in bbox_kwargs:
            fg_dirs = ps_results['ray_dirs_lock']
            bg_dirs = ps_results['ray_dirs_world']
            bg_dirs = torch.load('bg_dirs.pt').to(bg_dirs.device).to(bg_dirs.dtype)
        else:
            fg_dirs = ps_results['ray_dirs_object']
            bg_dirs = ps_results['ray_dirs_world']
        fg_coarse_rgbs, fg_coarse_sigmas = fg_mask_forward(wp=fg_wp,
                                                        dirs=fg_dirs,
                                                        pts=ps_results['fg_pts_object'],
                                                        mask=ps_results['ray_mask'],
                                                        lod=lod)
        bg_rgbs, bg_sigmas = bg_forward(
                                        network=self.bg_synthesis,
                                        wp=bg_wp,
                                        dirs=bg_dirs,
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
            fg_fine_rgbs, fg_fine_sigmas = fg_mask_forward(wp=fg_wp,
                                                        dirs=fine_dirs,
                                                        pts=fine_pts,
                                                        mask=ps_results['ray_mask'],
                                                        lod=lod)

            fg_rgbs = torch.cat((fg_fine_rgbs, fg_coarse_rgbs), dim=-2)
            fg_sigmas = torch.cat((fg_fine_sigmas, fg_coarse_sigmas), dim=-2)
            fg_pts_depth = torch.cat((fg_hs_results['pts_depth'], ps_results['fg_pts_depth']), dim=-2)
            # TODO maybe we can remove it
            _, indices = torch.sort(fg_pts_depth, dim=-2)
            rgbs = torch.gather(fg_rgbs, -2, indices.expand(-1, -1, -1, -1, -1, 3))
            sigmas = torch.gather(fg_sigmas, -2, indices)
            pts_depth = torch.gather(fg_pts_depth, -2, indices)
        else:
            fg_rgbs = fg_coarse_rgbs
            fg_sigmas = fg_coarse_sigmas
            fg_pts_depth = ps_results['fg_pts_depth']

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
        if foreground_only:
            if ps_kwargs.get('test_resolution', self.nerf_res) > 64:
                fg_render_results['rgb'] = F.interpolate(fg_render_results['rgb'].permute(0, 3, 1, 2), size=64, mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                fg_render_results['last_weights'] = F.interpolate(fg_render_results['last_weights'].permute(0, 3, 1, 2), size=64, mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            blend_rgb = fg_render_results['rgb'] + 0*(fg_render_results['last_weights']) 
        elif background_only:
            if self.use_bg_up:
                bk_feature_image = bg_render_results['rgb']
                bk_feature_image = bk_feature_image.permute(0, 3, 1, 2)
                bk_rgb_image = bk_feature_image[:, :3]
                bk_image_results = self.bg_superresolution(x=bk_feature_image,
                                                           rgb=bk_rgb_image,
                                                           wp=bg_wp[:, self.bg_synthesis_layers+1:],
                                                           lod=None,
                                                           noise_mode=noise_mode,
                                                           fused_modulate=fused_modulate,
                                                           impl=impl,
                                                           fp16_res=fp16_res)
                # TODO consider feature scale? sigmoid???
                bg_image = bk_image_results['image'].contiguous().permute(0, 2,3 ,1)
                bg_up_feature = bk_image_results['feature'].contiguous().permute(0, 2, 3, 1)
                bg_up_feature[...,:3] = bg_image
                blend_rgb = bg_up_feature
            else:
                blend_rgb = bg_render_results['rgb']
        else:
            if self.static_background:
                fg_image = fg_render_results['rgb']
                bk_image = self.bk_image.to(fg_image.dtype).to(fg_image.device)
                if bk_image.shape[1:3] != fg_image.shape[1:3]:
                    bk_image = bk_image.permute(0, 3, 1, 2)
                    bk_image = F.interpolate(bk_image, fg_image.shape[1:3], mode='bilinear').permute(0, 2, 3, 1)
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
                    # TODO consider feature scale? sigmoid???
                    bg_image = bk_image_results['image'].contiguous().permute(0, 2,3 ,1)
                    bg_up_feature = bk_image_results['feature'].contiguous().permute(0, 2, 3, 1)
                    bg_up_feature[...,:3] = bg_image
                    if ps_kwargs.get('test_resolution', self.nerf_res) > 64:
                        fg_render_results['rgb'] = F.interpolate(fg_render_results['rgb'].permute(0, 3, 1, 2), size=64, mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                        fg_render_results['last_weights'] = F.interpolate(fg_render_results['last_weights'].permute(0, 3, 1, 2), size=64, mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                    blend_rgb = fg_render_results['rgb']+ fg_render_results['last_weights'] * bg_up_feature
                else:
                    weights_map = 1 - fg_render_results['last_weights'].permute(0,3,1,2)
                    if ps_kwargs.get('test_resolution', self.nerf_res) > 64:
                        fg_render_results['rgb'] = F.interpolate(fg_render_results['rgb'].permute(0, 3, 1, 2), size=64, mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                        fg_render_results['last_weights'] = F.interpolate(fg_render_results['last_weights'].permute(0, 3, 1, 2), size=64, mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

                    fg_render_results['last_weights'] = F.interpolate(fg_render_results['last_weights'].permute(0, 3, 1, 2), size=128, mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                    fg_render_results['last_weights'] = F.interpolate(fg_render_results['last_weights'].permute(0, 3, 1, 2), size=64, mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                    blend_rgb = fg_render_results['rgb'] + fg_render_results['last_weights'] * bg_render_results['rgb']
        
        weights_map = 1 - fg_render_results['last_weights'].permute(0,3,1,2)
        weights = fg_render_results['weights']
        alphas = fg_render_results['alphas']
        pts_mid = fg_render_results['pts_mid']
        intervals = fg_render_results['deltas']

        # output
        # TODO Check value of feature range
        feature_image = blend_rgb.permute(0, 3, 1, 2)
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
        results = {**mapping_results, 
                   'wp':wp,  
                   'image': image, 
                   'image_raw': rgb_image, 
                   'latent': z, 
                   'camera': camera,
                   'avg_weights': avg_weights.reshape(-1, 1), 
                   'ray_mask': ps_results['ray_mask'], 
                   'weights_map': weights_map, 
                   'weights': weights, 
                   'alphas':alphas, 
                   'pts_mid': pts_mid, 
                   'intervals': intervals}
        if self.use_mask:
            ih, iw = image.shape[2:]
            wh, ww = weights_map.shape[2:] 
            up_weight_map = F.interpolate(weights_map, size=(ih, iw), mode='bilinear', align_corners=False)
            norm_up_weight_map = 2 * up_weight_map - 1
            results.update(norm_up_weight_map=norm_up_weight_map)
        if self.use_object:
            if 'g_image_bbox' in bbox_kwargs:
                results.update(image_bbox=bbox_kwargs['g_image_bbox'])
        return results

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


def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0]]], dtype=torch.float32)

def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.
    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]

def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None, mask=None):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    NP, M, _ = coordinates.shape
    # plane_features = plane_features.view(N*n_planes, C, H, W)

    coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds

    projected_coordinates = project_onto_planes(plane_axes, coordinates)
    projected_coordinates = projected_coordinates.reshape((coordinates.shape[0], 3,) + projected_coordinates.shape[-2:])
    
    # num_p, 3, 12, 2 --> 3, num_p, 12, 1, 3
    mask = mask.reshape((N, )+ mask.shape[-2:])
    b_idx = torch.nonzero(mask)[:, 0:1]
    b_idx = b_idx.unsqueeze(-1).unsqueeze(-1)
    b_idx = b_idx.expand(-1, n_planes, M, -1)
    b_idx = 2 * (b_idx + 0.5)/N - 1
    # TODO Normalize coordinates
    new_coordinates = torch.cat([projected_coordinates, b_idx], dim=-1)
    new_coordinates = new_coordinates.permute(1, 0, 2, 3).unsqueeze(-2)

    # bs, 3, c, h, w  --> 3, c, bs, h, w
    plane_features = plane_features.permute(1, 2, 0, 3, 4)
    output_features = torch.nn.functional.grid_sample(plane_features, new_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False)
    output_features = output_features.permute(2, 0, 1, 3, 4).reshape(NP, n_planes, M, C)
    return output_features



class SynthesisNetwork(nn.Module):

    def __init__(self,
                 w_dim,
                 in_channels,
                 num_layers,
                 out_channels,
                 use_dirs=True,
                 feature_dim=3,
                 eps=1e-8,
                 mlp_type='modulatemlp',
                 embed_cfg=dict(),
                 triplane_cfg=None,
                 hidden_channels=None
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.w_dim = w_dim
        self.out_channels = out_channels
        self.use_dirs = use_dirs
        self.feature_dim = feature_dim

        self.register_buffer('lod', torch.zeros(()))

        self.embedder = Embedder(**embed_cfg)
        self.triplane_cfg = triplane_cfg
        self.mlp_type = mlp_type
        self.use_triplane = self.triplane_cfg is not None
        if self.use_triplane:
            self.backbone = StyleGAN2Generator(**triplane_cfg) 
            num_layers = 1
            self.plane_axes = generate_planes()

        network = []
        film_kwargs = dict(add_bias=True,
                           demodulate=True,
                           use_wscale=True,
                           wscale_gain=1.0,
                           lr_mul=1.0,
                           activation_type='lrelu',
                           conv_clamp=None,
                           eps=1e-8)
        if not self.use_triplane:
            in_channels = self.embedder.out_dim
        
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
        if mlp_type == 'mlp':
            film_kwargs = dict(add_bias=True,
                               init_bias=0.0,
                               activation_type='softplus',
                               lr_mul=1,
                               use_wscale=True,
                               wscale_gain=1)
        arch_dict = {'modulatemlp': ModulateMLP,
                     'mlp': DenseLayer}
        c_out_channels = out_channels if hidden_channels is None else hidden_channels
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            out_channels = c_out_channels
            # film = FiLMLayer(in_channels, out_channels, w_dim)
            MLP = arch_dict.get(mlp_type)
            if mlp_type == 'modulatemlp':
                film = MLP(in_channels, out_channels, w_dim, **film_kwargs)
            elif mlp_type == 'mlp':
                film = MLP(in_channels, out_channels, **film_kwargs)
            else:
                raise NotImplementedError

            network.append(film)
        self.mlp_network = nn.Sequential(*network)
        if self.mlp_type == 'modulatemlp':
            if self.use_dirs:
                self.color_film = ModulateMLP(c_out_channels + 3, out_channels, w_dim, **film_kwargs)
            else:
                self.color_film = ModulateMLP(c_out_channels, out_channels, w_dim, **film_kwargs)

        self.sigma_head  = DenseLayer(in_channels=c_out_channels,
                                           out_channels=1,
                                           add_bias=True,
                                           init_bias=0.0,
                                           use_wscale=True,
                                           wscale_gain=1,
                                           lr_mul=1,
                                           activation_type='linear')
        self.color_head =  DenseLayer(in_channels=c_out_channels,
                                             out_channels=feature_dim,
                                             add_bias=True,
                                             init_bias=0.0,
                                             use_wscale=True,
                                             wscale_gain=1,
                                             lr_mul=1,
                                             activation_type='linear')

    def init_weights(self, ):
        pass

    def forward(self, wp, pts, dirs, lod=None, mask=None, noise_mode='random', box_warp=2):
        num_dims = pts.ndim
        assert num_dims in [3, 4, 5, 6]
        if num_dims == 5:
            bs, H, W, D, C = pts.shape
            dirs = dirs.unsqueeze(-2).expand_as(pts)
            pts = pts.reshape(bs, H * W * D, C)
            dirs = dirs.reshape(bs, H * W * D, C)
        elif num_dims == 4:
            bs, N, D, C = pts.shape
            dirs = dirs.unsqueeze(-2).expand_as(pts)
            pts = pts.reshape(bs, N * D, C)
            dirs = dirs.reshape(bs, N * D, C)
        elif num_dims == 3:
            dirs = dirs.unsqueeze(-2).expand_as(pts)
        elif num_dims == 6:
            bs, _, H, W, D, C = pts.shape
            dirs = dirs.unsqueeze(1).unsqueeze(-2).expand_as(pts)
            pts = pts.reshape(bs, H * W * D, C)
            dirs = dirs.reshape(bs, H * W * D, C)

        # x = self.gridwarper(pts)
        if not self.use_triplane:
            x = self.embedder(pts)
        else:
            bs, num_bbox = wp.shape[:2]
            wp = wp.reshape((-1, )+wp.shape[2:])
            plane_features = self.backbone.synthesis(wp,
                                        noise_mode=noise_mode,
                                        fused_modulate=False,
                                        impl='cuda',
                                        fp16_res=None)['image']
            N, C, H, W = plane_features.shape
            plane_features = plane_features.reshape(N, 3, -1, H, W)
            plane_axes = self.plane_axes.to(plane_features.device)
            sample_features = sample_from_planes(plane_axes,
                                                 plane_features,
                                                 pts,
                                                 mode='bilinear',
                                                 padding_mode='zeros',
                                                 box_warp=box_warp,
                                                 mask=mask)
            x = sample_features.mean(1)
        bs, L, _ = x.shape
        for idx, layer in enumerate(self.mlp_network):
            if self.mlp_type == 'modulatemlp': 
                if mask is not None:
                    x = layer(x, wp[:, :, idx], mask=mask)
                else:
                    x = layer(x, wp[:, idx], mask=mask)
            elif self.mlp_type == 'mlp':
                x = x.reshape(bs*L, -1)
                x = layer(x)
                x = x.reshape(bs, L, -1)
            else:
                raise NotImplementedError
        C = x.shape[-1]
        sigma = self.sigma_head(x.reshape(bs*L, C)).reshape(bs, L, -1)
        if self.use_dirs:
            dirs = torch.cat([x, dirs], dim=-1)
        else:
            dirs = x
    
        if self.mlp_type == 'modulatemlp': 
            if mask is not None:
                color = self.color_film(dirs, wp[:, :, len(self.mlp_network)], mask=mask)
            else:
                color = self.color_film(dirs, wp[:, len(self.mlp_network)], mask=mask)
        elif self.mlp_type == 'mlp':
            color = dirs
        else:
            raise NotImplementedError

        bs, L, C = color.shape
        color = self.color_head(color.reshape(bs*L, C)).reshape(bs, L, -1)

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
                else:
                    block = getattr(self, f'block{block_idx}')
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
                rgb = cur_rgb
            elif cur_lod < lod < cur_lod + 1:
                alpha = np.ceil(lod) - lod
                # TODO bilinear?
                rgb = F.interpolate(rgb, scale_factor=2, **up_kwargs)
                rgb = cur_rgb * alpha + rgb * (1 - alpha)
            elif lod >= cur_lod + 1:
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

class ModulateMLP(nn.Module):
    """Implements the convolutional layer with style modulation."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 # resolution,
                 w_dim,
                 # kernel_size,
                 add_bias,
                 # scale_factor,
                 # filter_kernel,
                 demodulate,
                 use_wscale,
                 wscale_gain,
                 lr_mul,
                 # noise_type,
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
        # self.resolution = resolution
        self.w_dim = w_dim
        # self.kernel_size = kernel_size
        self.add_bias = add_bias
        # self.scale_factor = scale_factor
        # self.filter_kernel = filter_kernel
        self.demodulate = demodulate
        self.use_wscale = use_wscale
        self.wscale_gain = wscale_gain
        self.lr_mul = lr_mul
        # self.noise_type = noise_type.lower()
        self.activation_type = activation_type
        self.conv_clamp = conv_clamp
        self.eps = eps

        self.space_of_latent = 'W'

        # Set up weight.
        weight_shape = (out_channels, in_channels, 1, 1)
        fan_in = 1 * 1 * in_channels
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
    def extra_repr(self):
        return (f'in_ch={self.in_channels}, '
                f'out_ch={self.out_channels}, '
                f'wscale_gain={self.wscale_gain:.3f}, '
                f'bias={self.add_bias}, '
                f'lr_mul={self.lr_mul:.3f}, '
                f'demodulate={self.demodulate}, '
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
            # if w.ndim != 2 or w.shape[1] != self.w_dim:
            #     raise ValueError(f'The input tensor should be with shape '
            #                      f'[batch_size, w_dim], where '
            #                      f'`w_dim` equals to {self.w_dim}!\n'
            #                      f'But `{w.shape}` is received!')
            if w.ndim == 3:
                bs, n, c = w.shape
                style = self.style(w.reshape(bs*n, c), impl=impl).reshape(bs, n, -1)
            elif w.ndim == 2:
                style = self.style(w, impl=impl)
            else:
                raise ValueError('w.ndim must be in [2,3]')
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
                fused_modulate=False,
                impl='cuda',
                mask=None):

        if x.ndim == 3: 
            x = x[..., None]
            x = x.permute(0, 2, 1, 3)
        else:
            raise NotImplementedError

        if mask is not None:
            index = torch.nonzero(mask)

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
            style_dim = style.ndim
            if style_dim == 3:
                bs, n, c = style.shape
                style = style.reshape(bs*n, c)
                N = bs*n
            if not self.demodulate:
                _style = style * self.wscale  # Equivalent to scaling weight.
            else:
                _style = style

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
                if style_dim == 3:
                    _style = _style.to(dtype).reshape(bs, n, in_ch, 1, 1)
                    s_style = _style[index[:, 0], index[:, 1]]
                    x = x * s_style
                    # _weight = _weight.reshape((bs,n)+_weight.shape[1:]).to(dtype)
                    # w = _weight[index[:, 0], index[:, 1]]
                    w = weight.to(dtype)
                else:
                    x = x * _style.to(dtype).reshape(N, in_ch, 1, 1)
                    w = weight.to(dtype)
            groups = 1
        else:  # Use group convolution to fuse style modulation and convolution.
            x = x.reshape(1, N * in_ch, H, W)
            w = _weight.reshape(N * out_ch, in_ch, kh, kw).to(dtype)
            groups = N

        up = 1
        padding = 1 // 2
        x = conv2d_gradfix.conv2d(
            x, w, stride=1, padding=padding, groups=groups, impl=impl)

        if not fused_modulate:
            if self.demodulate:
                if style_dim==3:
                    decoef = decoef.to(dtype).reshape(bs, n, out_ch, 1, 1)
                    decoef = decoef[index[:, 0], index[:, 1]]
                else:
                    decoef = decoef.to(dtype).reshape(N, out_ch, 1, 1)
            noise = None
            if self.demodulate and noise is not None:
                raise NotImplementedError
            else:
                if self.demodulate:
                    x = x * decoef
        else:
            x = x.reshape(N, out_ch, H * up, W * up)

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
        x = x.permute(0, 2, 1, 3)
        x = x.squeeze(-1)
        assert x.dtype == dtype
        if use_adain and False:
            assert style.dtype == torch.float32
            return x, style
        else:
            return x
