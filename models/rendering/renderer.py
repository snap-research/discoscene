# python3.7
"""Contains the function to integrate along the camera ray."""

from sys import intern
import torch
import torch.nn.functional as F

__all__ = ['Renderer', 'renderer']

_CLAMP_MODE = ['softplus', 'relu', 'mipnerf']
_FILL_MODE = ['debug', 'weight']

class Renderer(object):
    """Integrate the values along the ray.
    

    """
    def __init__(self,
                 last_back=False,
                 white_back=False,
                 clamp_mode=None,
                 render_mode=None,
                 fill_mode=None,
                 max_depth=None,
                 num_per_group=None):
        """Initializes with basic settings."""
        self.last_back = last_back
        self.white_back = white_back
        self.clamp_mode = clamp_mode
        self.render_mode = render_mode
        self.fill_mode = fill_mode
        self.max_depth = max_depth
        self.num_per_group = num_per_group

    def __call__(self, rgbs, sigmas, pts_z, noise_std, **kwargs):
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

        return renderer(rgbs=rgbs,
                           sigmas=sigmas,
                           pts_z=pts_z,
                           noise_std=noise_std,
                           last_back=self.last_back,
                           white_back=self.white_back,
                           clamp_mode=self.clamp_mode,
                           render_mode=self.render_mode,
                           fill_mode=self.fill_mode,
                           max_depth=self.max_depth,
                           num_per_group=self.num_per_group)

def renderer(rgbs,
                sigmas,
                pts_z,
                noise_std,
                last_back=False,
                white_back=False,
                clamp_mode=None,
                render_mode=None,
                fill_mode=None,
                max_depth=None,
                num_per_group=None):
    """ Integrate the values along the ray.

    Args:
        rgbs: （batch_size, H, W, num_steps, 3) or (batch_size, num_rays, num_steps, 3)
        sigmas: （batch_size, H, W, num_steps, 1) or (batch_size, num_rays, num_steps, 1)
        pts_z: (batch_size, H, W, num_steps, 1) or (batch_size, num_rays, num_steps, 1)
        noise_std: 
    
    Returns:
        rgb: (batch_size, H, W, 3) or (batch_size, num_rays, 3)
        depth: (batch_size, H, W, 1) or (batch_size, num_rays, 1)
        weights: (batch_size, H, W, num_steps, 1) or (batch_size, num_rays, num_steps, 1)
        alphas: (batch_size, H, W, 1) or (batch_size, num_rays, 1)
    """
    num_dims = rgbs.ndim
    assert num_dims in [4, 5]
    assert num_dims == sigmas.ndim and num_dims == pts_z.ndim

    if num_dims == 4:
        batch_size, num_rays, num_steps = rgbs.shape[:3]
    else:
        batch_size, H, W, num_steps = rgbs.shape[:4]
        rgbs = rgbs.reshape(batch_size, H * W, num_steps, rgbs.shape[-1])
        sigmas = sigmas.reshape(batch_size, H * W, num_steps, sigmas.shape[-1])
        pts_z = pts_z.reshape(batch_size, H * W, num_steps, pts_z.shape[-1])

    if num_per_group is None:
        num_per_group = num_steps
    assert num_steps % num_per_group== 0
    # Get deltas for rendering.
    deltas = pts_z[:, :, 1:] - pts_z[:, :, :-1]
    # if max_depth is not None:
    #     delta_inf = max_depth - pts_z[:, :, -1:]
    # else:
    # print(max_depth)
    if max_depth is None:
        max_depth = 1e10
    delta_inf = max_depth * torch.ones_like(deltas[:, :, :1])
    deltas = torch.cat([deltas, delta_inf], -2)
    if render_mode == 'no_dist':
        deltas[:] = 1
    
    # Get alpha
    noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std
    if clamp_mode == 'softplus':
        alphas = 1-torch.exp(-deltas * (F.softplus(sigmas + noise)))
    elif clamp_mode == 'relu':
        alphas = 1 - torch.exp(-deltas * (F.relu(sigmas + noise)))
    elif clamp_mode == 'mipnerf':
        alphas = 1 - torch.exp(-deltas * (F.softplus(sigmas + noise - 1)))
    else:
        raise ValueError(f'Invalid clamping mode: `{clamp_mode}`!\n'
                        f'Types allowed: {list(_CLAMP_MODE)}.')

    # Get accumulated alphas
    all_alphas_shifted = torch.cat([torch.ones_like(alphas[:, :, :1]), 1 - alphas + 1e-10], -2)
    all_cum_alphas_shifted = torch.cumprod(all_alphas_shifted, -2)[:, :, :-1]
    group_cum_alphas_shifted = all_cum_alphas_shifted[:, :, ::num_per_group]
    alphas = alphas.reshape(alphas.shape[:2] + (num_steps//num_per_group, num_per_group, alphas.shape[-1])) 
    alphas_shifted = torch.cat([torch.ones_like(alphas[:, :, :, :1]), 1 - alphas + 1e-10], -2) 
    cum_alphas_shifted = torch.cumprod(alphas_shifted, -2)

    # Get weights
    weights = alphas * cum_alphas_shifted[:, :, :, :-1]
    weights_sum = weights.sum(3)

    rgbs = rgbs.reshape(rgbs.shape[:2] + (num_steps//num_per_group, num_per_group, rgbs.shape[-1])) 
    pts_z = pts_z.reshape(pts_z.shape[:2] + (num_steps//num_per_group, num_per_group, pts_z.shape[-1])) 

    if last_back:
        weights[:, :, :, -1] += (1 - weights_sum)
    
    # Get rgb and depth
    rgb_final = torch.sum(weights * rgbs, -2)
    depth_final = torch.sum(weights * pts_z, -2)

    if white_back:
        rgb_final = rgb_final + 1 - weights_sum

    if fill_mode == 'debug':
        rgb_final[weights_sum.squeeze(-1) < 0.9] = torch.tensor([1., 0, 0, 0], device=rgbs.device)
    elif fill_mode == 'weight':
        rgb_final = weights_sum.expand_as(rgb_final)

    weights_final = weights.reshape(alphas.shape[:2]+(num_steps, 1))
    if num_dims == 5:
        rgb_final = rgb_final.reshape(batch_size, H, W, rgbs.shape[-1])
        depth_final = depth_final.reshape(batch_size, H, W, 1)
        weights_final = weights_final.reshape(batch_size, H, W, num_steps, 1)
        group_cum_alphas_shifted = group_cum_alphas_shifted.reshape(batch_size, H, W, 1)
    else:
        rgb_final = rgb_final.reshape(batch_size, num_rays, rgbs.shape[-1])
        depth_final = depth_final.reshape(batch_size, num_rays, 1)
        weights_final = weights_final.reshape(batch_size, num_rays, num_steps, 1)
        group_cum_alphas_shifted = group_cum_alphas_shifted.reshape(batch_size, num_rays, 1)
        
    results = {
        'rgb': rgb_final,
        'depth': depth_final,
        'weights': weights_final,
        'alphas': alphas,
        'deltas': deltas,
    }

    return results


if __name__ == '__main__':
    integrator = Renderer(last_back=False,
                 white_back=False,
                 clamp_mode='relu',
                 render_mode=None,
                 fill_mode=None,
                 max_depth=None,
                 num_per_group=None)

    device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    H = W = 64
    num_rays = 64*64
    num_steps = 24

    # coarse_rgbs = torch.arange((batch_size, num_rays, num_steps, 3), device=device)
    # coarse_sigmas = torch.ones((batch_size, num_rays, num_steps, 1), device=device)
    coarse_rgbs = 1+torch.arange(0, num_steps).reshape(1,  1, num_steps, 1).to(device)
    coarse_rgbs = coarse_rgbs.repeat(batch_size, num_rays, 1, 1).float()

    coarse_sigmas = torch.arange(0, num_steps).reshape(1, 1, num_steps, 1).to(device)
    coarse_sigmas = coarse_sigmas.repeat(batch_size, num_rays, 1, 1).float()

    pts_z = 1+0.2*torch.arange(0, num_steps).reshape(1, 1, num_steps, 1).to(device)
    pts_z = pts_z.repeat(batch_size, num_rays, 1, 1).float()

    results = integrator(rgbs=coarse_rgbs,
                         sigmas=coarse_sigmas,
                         pts_z=pts_z,
                         max_depth=1e-3,
                         noise_std=0)

    print(results['rgb'].shape)
    print(results['depth'].shape)
    print(results['weights'].shape)
    print(results['alphas'].shape)
    print(results['deltas'][0,100])
    print(results['alphas'][0,100])
    print(results['rgb'][0,100:110])
    print(results['weights'][0,100,:])
