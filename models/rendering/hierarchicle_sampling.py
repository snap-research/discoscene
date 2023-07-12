# python3.7
"""Contains the function to sample the points in 3D space."""

import torch

from .renderer import renderer

__all__ = ['HierarchicalSampling']

class HierarchicalSampling(object):
    """Hierarchically samples the points according to the coarse results.

    Args:
        last_back:
        white_back:
        clamp_mode:
        render_mode:
        fill_mode:
        max_depth:
        num_per_group:
    
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

    def __call__(self, coarse_rgbs, coarse_sigmas, pts_z, ray_origins, ray_dirs, noise_std=0.5, **kwargs):
        """
        Args:
            coarse_rgbs: (batch_size, num_of_rays, num_steps, 3) or (batch_size, H, W, num_steps, 3)
            coarse_sigmas: (batch_size, num_of_rays, num_steps, 1) or (batch_size, H, W, num_steps, 1)
            pts_z: (batch_size, num_of_rays, num_steps, 1) or (batch_size, H, W, num_steps, 1)
            ray_origins: (batch_size, num_of_rays, 3) or (batch_size, H, W, 3)
            ray_dirs: Ray directions. (batch_size, num_of_rays, 3) or (batch_size, H, W, 3)
            noise_std: 

        Returns: dict()
            pts: (batch_size, num_of_rays, num_steps, 3) or (batch_size, H, W, num_steps, 3)
            pts_z: (batch_size, num_of_rays, num_steps, 1) or (batch_size, H, W, num_steps, 1)

        """

        for key, value in kwargs.items():
            if key in self.__dict__.keys():
                self.__dict__[key] = value
        
        num_dims = coarse_rgbs.ndim
        assert coarse_rgbs.ndim in [4, 5] 
        assert coarse_sigmas.ndim == coarse_rgbs.ndim
        assert pts_z.ndim == coarse_rgbs.ndim
        assert ray_origins.ndim == pts_z.ndim - 1 and ray_dirs.ndim == pts_z.ndim - 1

        if num_dims == 4:
            batch_size, num_rays, num_steps = coarse_rgbs.shape[:3]
        else:
            batch_size, H, W, num_steps = coarse_rgbs.shape[:4]
            num_rays = H * W
            coarse_rgbs = coarse_rgbs.reshape(batch_size, num_rays, num_steps, coarse_rgbs.shape[-1])
            coarse_sigmas = coarse_sigmas.reshape(batch_size, num_rays, num_steps, coarse_sigmas.shape[-1])
            pts_z = pts_z.reshape(batch_size, num_rays, num_steps, pts_z.shape[-1])
            ray_origins = ray_origins.reshape(batch_size, num_rays, -1)
            ray_dirs = ray_dirs.reshape(batch_size, num_rays, -1)

        # Get the importance of all points
        renderer_results = renderer(rgbs=coarse_rgbs,
                                          sigmas=coarse_sigmas,
                                          pts_z=pts_z,
                                          noise_std=noise_std,
                                          last_back=self.last_back,
                                          white_back=self.white_back,
                                          clamp_mode=self.clamp_mode,
                                          render_mode=self.render_mode,
                                          fill_mode=self.fill_mode,
                                          max_depth=self.max_depth,
                                          num_per_group=self.num_per_group)
        weights = renderer_results['weights'].reshape(batch_size * num_rays, num_steps) + 1e-5

        # Importance sampling
        pts_z = pts_z.reshape(batch_size * num_rays, num_steps)
        pts_z_mid = 0.5 * (pts_z[:, :-1] + pts_z[:, 1:])
        fine_pts_z = self.sample_pdf(pts_z_mid,
                                      weights[:, 1:-1],
                                      num_steps,
                                      det=False).detach().reshape(batch_size, num_rays, num_steps, 1)
        fine_pts = ray_origins.unsqueeze(2).contiguous() + ray_dirs.unsqueeze(2).contiguous() * fine_pts_z.contiguous()

        if num_dims == 4: 
            fine_pts = fine_pts.reshape(batch_size, num_rays, num_steps, 3)                     
            fine_pts_z = fine_pts_z.reshape(batch_size, num_rays, num_steps, 1)
            ray_dirs = ray_dirs.reshape(batch_size, num_rays, 3)
        else:
            fine_pts = fine_pts.reshape(batch_size, H, W, num_steps, 3) 
            fine_pts_z = fine_pts_z.reshape(batch_size, H, W, num_steps, 1)
            ray_dirs = ray_dirs.reshape(batch_size, H, W, 3)
        results = {
            'pts': fine_pts,
            'pts_z': fine_pts_z,
            'ray_dirs': ray_dirs,
        }

        return results



    def sample_pdf(self, bins, weights, N_importance, det=False, eps=1e-5):
        """Sample @N_importance samples from @bins with distribution defined by @weights.

        Args:
            bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
            weights: (N_rays, N_samples_)
            N_importance: the number of samples to draw from the distribution
            det: deterministic or not
            eps: a small number to prevent division by zero

        Returns:
            samples: the sampled samples
        Source: https://github.com/kwea123/nerf_pl/blob/master/models/rendering.py

        """
        N_rays, N_samples_ = weights.shape
        weights = weights + eps  # prevent division by zero (don't do inplace op!)
        pdf = weights / torch.sum(weights, -1,
                                keepdim=True)  # (N_rays, N_samples_)
        cdf = torch.cumsum(
            pdf, -1)  # (N_rays, N_samples), cumulative distribution function
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf],
                        -1)  # (N_rays, N_samples_+1)
        # padded to 0~1 inclusive

        if det:
            u = torch.linspace(0, 1, N_importance, device=bins.device)
            u = u.expand(N_rays, N_importance)
        else:
            u = torch.rand(N_rays, N_importance, device=bins.device)
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u)
        below = torch.clamp_min(inds - 1, 0)
        above = torch.clamp_max(inds, N_samples_)

        inds_sampled = torch.stack([below, above],
                                -1).view(N_rays, 2 * N_importance)
        cdf_g = torch.gather(cdf, 1, inds_sampled)
        cdf_g = cdf_g.view(N_rays, N_importance, 2)
        bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom[denom < eps] = 1  # denom equals 0 means a bin has weight 0, in which case it will not be sampled
        # anyway, therefore any value for it is fine (set to 1 here)

        samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1] -
                                                                bins_g[..., 0])
        return samples


if __name__ == '__main__':
    hiesampler = HierarchicalSampling(last_back=False,
                 white_back=False,
                 clamp_mode='softplus',
                 render_mode=None,
                 fill_mode='weight',
                 max_depth=None,
                 num_per_group=None)
    device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    H = W = 64
    num_rays = 64*64
    num_steps = 24

    coarse_rgbs = torch.rand((batch_size, num_rays, num_steps, 3), device=device)
    coarse_sigmas = torch.rand((batch_size, num_rays, num_steps, 1), device=device)
    pts_z = torch.rand((batch_size, num_rays, num_steps, 1), device=device)
    ray_origins = torch.rand((batch_size, num_rays, 3), device=device)
    ray_dirs = torch.rand((batch_size, num_rays, 3), device=device)

    results = hiesampler(coarse_rgbs=coarse_rgbs,
                         coarse_sigmas=coarse_sigmas,
                         pts_z=pts_z,
                         ray_origins=ray_origins,
                         ray_dirs=ray_dirs)

    print(results['pts'][0,0,0])
    print(results['pts'][0,0,1])
    print(results['pts'][0,0,2])
    print(results['pts_z'][0,0,0])

