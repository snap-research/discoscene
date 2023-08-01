# python3.7
"""Contains the function to integrate along the camera ray."""

from sys import intern
import torch
import torch.nn.functional as F

__all__ = ['RendererBbox', 'renderer_bbox']

_CLAMP_MODE = ['softplus', 'relu', 'mipnerf']
_FILL_MODE = ['debug', 'weight']

class RendererBbox(object):
    """Integrate the values along the ray.
    

    """
    def __init__(self,
                 last_back=False,
                 white_back=False,
                 clamp_mode=None,
                 render_mode=None,
                 fill_mode=None,
                 max_depth=1e10,
                 num_per_group=None):
        """Initializes with basic settings."""
        self.last_back = last_back
        self.white_back = white_back
        self.clamp_mode = clamp_mode
        self.render_mode = render_mode
        self.fill_mode = fill_mode
        self.max_depth = max_depth
        self.num_per_group = num_per_group

    def __call__(self, rgbs, sigmas, pts_z, noise_std, ray_mask=None, **kwargs):
        """
        Args:
            rgbs: （batch_size, H, W, num_steps, 3) or (batch_size, num_rays, num_steps, 3)
            sigmas: （batch_size, H, W, num_steps, 1) or (batch_size, num_rays, num_steps, 1)
            pts_z: (batch_size, H, W, num_steps, 1) or (batch_size, num_rays, num_steps, 1)
            noise_std: 

        Returns:
        """

        for key, value in kwargs.items():
            if key in self.__dict__.keys():
                self.__dict__[key] = value

        return renderer_bbox(rgbs=rgbs,
                           sigmas=sigmas,
                           pts_z=pts_z,
                           noise_std=noise_std,
                           last_back=self.last_back,
                           white_back=self.white_back,
                           clamp_mode=self.clamp_mode,
                           render_mode=self.render_mode,
                           fill_mode=self.fill_mode,
                           max_depth=self.max_depth,
                           num_per_group=self.num_per_group,
                           ray_mask=ray_mask)

def renderer_bbox(rgbs,
                sigmas,
                pts_z,
                noise_std,
                last_back=False,
                white_back=False,
                clamp_mode=None,
                render_mode=None,
                fill_mode=None,
                max_depth=None,
                last_dist=None,
                num_per_group=None,
                ray_mask=None):
    """ Integrate the values along the ray.

    Args:
        rgbs: （batch_size, N, H, W, num_steps, 3) or (batch_size, num_rays, num_steps, 3)
        sigmas: （batch_size, N,  H, W, num_steps, 1) or (batch_size, num_rays, num_steps, 1)
        pts_z: (batch_size, N, H, W, num_steps, 1) or (batch_size, num_rays, num_steps, 1)
        noise_std: 
    
    Returns:
        rgb: (batch_size, H, W, 3) or (batch_size, num_rays, 3)
        depth: (batch_size, H, W, 1) or (batch_size, num_rays, 1)
        weights: (batch_size, H, W, num_steps, 1) or (batch_size, num_rays, num_steps, 1)
        alphas: (batch_size, H, W, 1) or (batch_size, num_rays, 1)
    """
    num_dims = rgbs.ndim
    assert num_dims in [5, 6]
    assert num_dims == sigmas.ndim and num_dims == pts_z.ndim

    if num_dims == 5:
        batch_size, num_objects, num_rays, num_steps = rgbs.shape[:4]
    else:
        batch_size, num_objects, H, W, num_steps = rgbs.shape[:5]
        rgbs = rgbs.reshape(batch_size, num_objects, H * W, num_steps, rgbs.shape[-1])
        sigmas = sigmas.reshape(batch_size, num_objects, H * W, num_steps, sigmas.shape[-1])
        pts_z = pts_z.reshape(batch_size, num_objects, H * W, num_steps, pts_z.shape[-1])
        num_rays = H*W
    if num_per_group is None:
        num_per_group = num_steps
    assert num_steps % num_per_group== 0

    # TODO 
    # if max_depth < 10:
    #    sigmas[sigmas<torch.quantile(sigmas, 0.5, dim=-2).unsqueeze(-2)] = -10
    # stack N points
    sigmas = sigmas.permute(0, 2, 1, 3, 4).reshape(batch_size, num_rays, num_objects*num_steps, sigmas.shape[-1])
    rgbs = rgbs.permute(0, 2, 1, 3, 4).reshape(batch_size, num_rays, num_objects*num_steps, rgbs.shape[-1])
    pts_z = pts_z.permute(0, 2, 1, 3, 4).reshape(batch_size, num_rays, num_objects*num_steps, pts_z.shape[-1])

    pts_z_sorted, idx_sorted = torch.sort(pts_z, -2)
    sigmas_sorted = torch.gather(sigmas, dim=-2, index=idx_sorted)
    rgbs_sorted = torch.gather(rgbs, dim=-2, index=idx_sorted.expand_as(rgbs))
     
    # Get deltas for rendering.
    deltas = pts_z_sorted[:, :, 1:] - pts_z_sorted[:, :, :-1]
    # if max_depth is not None:
    #     delta_inf = max_depth - pts_z[:, :, -1:]
    delta_inf = max_depth * torch.ones_like(deltas[:, :, :1])
    deltas = torch.cat([deltas, delta_inf], -2)
    if render_mode == 'no_dist':
        deltas[:] = 1
    
    # Get alpha
    noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std
    if clamp_mode == 'softplus':
        alphas = 1-torch.exp(-deltas * (F.softplus(sigmas_sorted + noise)))
    elif clamp_mode == 'relu':
        alphas = 1 - torch.exp(-deltas * (F.relu(sigmas_sorted + noise)))
    elif clamp_mode == 'mipnerf':
        alphas = 1 - torch.exp(-deltas * (F.softplus(sigmas_sorted + noise - 1)))
    else:
        raise ValueError(f'Invalid clamping mode: `{clamp_mode}`!\n'
                        f'Types allowed: {list(_CLAMP_MODE)}.')
    # set for bg
    # if max_depth > 0:
    #     alphas[:, :, -1] = 1 # bs, L, S, 1 

    # Get accumulated alphas
    alphas_shifted = torch.cat([torch.ones_like(alphas[:, :, :1]), 1 - alphas + 1e-10], -2) 
    cum_alphas_shifted = torch.cumprod(alphas_shifted, -2)[:, :, :-1]

    # Get weights
    weights = alphas * cum_alphas_shifted # bs, L, S, 1
    weights_sum = weights.sum(2) # bs, L,  1
    last_weights = 1 - weights_sum
    # rgbs = rgbs.reshape(rgbs.shape[:2] + (num_steps//num_per_group, num_per_group, rgbs.shape[-1])) 
    # pts_z = pts_z.reshape(pts_z.shape[:2] + (num_steps//num_per_group, num_per_group, pts_z.shape[-1])) 

    if last_back:
        weights[:, :, :, -1] += (1 - weights_sum)
    
    # Get rgb and depth
    rgb_final = torch.sum(weights * rgbs_sorted, -2)
    depth_final = torch.sum(weights * pts_z_sorted, -2)

    if white_back:
        rgb_final = rgb_final + 1 - weights_sum

    if fill_mode == 'debug':
        rgb_final[weights_sum.squeeze(-1) < 0.9] = torch.tensor([1., 0, 0, 0], device=rgbs.device)
    elif fill_mode == 'weight':
        rgb_final = weights_sum.expand_as(rgb_final)
    weights_final = weights.reshape(alphas.shape[:2]+(weights.shape[-2], 1))
    alphas_final = alphas.reshape(alphas.shape[:2]+(weights.shape[-2], 1))
    if ray_mask is not None:
        mask = (ray_mask.sum(dim=1, keepdim=False)>0).reshape(batch_size, num_rays)
        select_weight = torch.masked_select(weights_final, mask[...,None,None]).reshape(-1, num_objects*num_steps).sum(dim=-1)
        avg_weight = select_weight.mean()
    # re_sort to differnt objects
    _ , re_idx_sorted = torch.sort(idx_sorted, -2)
    object_weights = torch.gather(weights_final, dim=-2, index=re_idx_sorted)
    object_weights = object_weights.reshape(batch_size, num_rays, num_objects, num_steps, object_weights.shape[-1])
    object_weights = object_weights.permute(0, 2, 1, 3, 4) # batch_size, num_object, H, W, num_steps, 1
    
    object_alphas = torch.gather(alphas_final, dim=-2, index=re_idx_sorted)
    object_alphas = object_alphas.reshape(batch_size, num_rays, num_objects, num_steps, object_weights.shape[-1])
    object_alphas = object_alphas.permute(0, 2, 1, 3, 4)
    

    object_deltas = torch.gather(deltas, dim=-2, index=re_idx_sorted)
    object_deltas = object_deltas.reshape(batch_size, num_rays, num_objects, num_steps, object_weights.shape[-1])
    object_deltas = object_deltas.permute(0, 2, 1, 3, 4)

    pts_z = pts_z.reshape(batch_size, num_rays, num_objects, num_steps, pts_z.shape[-1])
    pts_z = pts_z.permute(0, 2, 1, 3, 4)
    pts_mid = pts_z + object_deltas/2
    # pts_mid = (pts_z[:, :, :, :-1] + pts_z[:, :, :, 1:])/2
    # pts_end = max_depth/2 * torch.ones_like(pts_z[:, :, :, :1])
    # pts_mid = torch.cat([pts_mid, pts_end], dim=3)

    if num_dims == 6:
        rgb_final = rgb_final.reshape(batch_size, H, W, rgbs.shape[-1])
        depth_final = depth_final.reshape(batch_size, H, W, 1)
        weights_final = object_weights.reshape(batch_size, num_objects, H, W, num_steps, 1)
        alphas_final = object_alphas.reshape(batch_size, num_objects, H, W, num_steps, 1)
        last_weights = last_weights.reshape(batch_size, H, W, 1)
        pts_mid = pts_mid.reshape(batch_size, num_objects, H, W, num_steps, pts_z.shape[-1])
        deltas_final = object_deltas.reshape(batch_size, num_objects, H, W, num_steps, 1)
    else:
        rgb_final = rgb_final.reshape(batch_size, num_rays, rgbs.shape[-1])
        depth_final = depth_final.reshape(batch_size, num_rays, 1)
        weights_final = object_weights.reshape(batch_size, num_objects, num_rays, num_steps, 1)
        alphas_final = object_alphas.reshape(batch_size, num_objects, num_rays, num_steps, 1)
        last_weights = last_weights.reshape(batch_size, num_rays, 1)
        pts_mid = pts_mid.reshaep(batch_size, num_rays, 1)
        deltas_final = object_deltas.reshape(batch_size, num_ray, 1)
        
    results = {
        'rgb': rgb_final,
        'depth': depth_final,
        'weights': weights_final,
        'alphas': alphas_final,
        'last_weights': last_weights,
        # 'alphas': alphas,
        'deltas': deltas_final,
        'pts_mid': pts_mid,
    }
    if ray_mask is not None:
        results.update(avg_weights=avg_weight)

    return results


if __name__ == '__main__':
    integrator = RendererBbox(last_back=False,
                 white_back=False,
                 clamp_mode='relu',
                 render_mode=None,
                 fill_mode=None,
                 num_per_group=None)

    device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    H = W = 64
    num_rays = 64*64
    num_steps = 24

    coarse_rgbs = 1+torch.arange(0, num_steps).reshape(1, 1, 1, num_steps, 1).to(device)
    coarse_rgbs = coarse_rgbs.repeat(batch_size, 1, num_rays, 1, 1).float()
    coarse_rgbs_ = torch.zeros_like(coarse_rgbs)
    coarse_rgbs = torch.cat([coarse_rgbs, coarse_rgbs_], dim=1)

    coarse_sigmas = torch.arange(0, num_steps).reshape(1, 1, 1, num_steps, 1).to(device)
    coarse_sigmas = coarse_sigmas.repeat(batch_size, 1, num_rays, 1, 1).float()
    coarse_sigmas_ = torch.ones_like(coarse_sigmas)*-100
    coarse_sigmas = torch.cat([coarse_sigmas, coarse_sigmas_], dim=1)

    pts_z = 1+0.2*torch.arange(0, num_steps).reshape(1, 1, 1, num_steps, 1).to(device)
    pts_z = pts_z.repeat(batch_size, 1, num_rays, 1, 1).float()
    pts_z_ = torch.zeros_like(pts_z)
    pts_z = torch.cat([pts_z, pts_z_], dim=1)

    results = integrator(rgbs=coarse_rgbs,
                         sigmas=coarse_sigmas,
                         max_depth=1e-3,
                         pts_z=pts_z,
                         noise_std=0)
    print(results['rgb'].shape)
    print(results['depth'].shape)
    print(results['weights'].shape)
    print(results['alphas'].shape)
    print(results['deltas'][0,100,24:])
    print(results['alphas'][0,100,:])
    print(results['rgb'][0,100:110])
    print(results['weights'][0,0,100,:])
