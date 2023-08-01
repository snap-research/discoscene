# python3.7
"""Contains the function to sample the points in 3D space."""

import torch
import numpy as np
import math
import random

from .utils import normalize_vecs
from .utils import truncated_normal

__all__ = ['PointsSampling', 'sample_pts_in_cam_coord', 'sample_cam_positions', 'trans_pts_cam2world', 'create_cam2world_matrix', 'perturb_points']
_CAMERA_DIST = ['uniform', 'normal', 'gaussian', 'hybrid', 'truncated_gaussian', 'spherial_uniform']

class PointsSampling(object):
    """Samples 3D points in the world space.

    Args:
        num_steps: The number of points to be sampled along one ray.
        ray_start: 
        ray_end:
        radius:
        horizontal_mean:
        horizontal_stddev:
        vertical_mean:
        vertical_stddev:
        camera_dist:
        fov:
        perturb_points:
        opencv_axis:
    """
    def __init__(self,
                 num_steps,
                 ray_start,
                 ray_end,
                 radius,
                 horizontal_mean,
                 horizontal_stddev,
                 vertical_mean,
                 vertical_stddev,
                 camera_dist,
                 fov,
                 perturb_mode,
                 opencv_axis=False,
                 ):
        """Initializes with basic settings."""
        self.num_steps = num_steps
        self.ray_start = ray_start
        self.ray_end = ray_end
        self.radius = radius
        self.horizontal_mean = horizontal_mean
        self.horizontal_stddev = horizontal_stddev
        self.vertical_mean = vertical_mean
        self.vertical_stddev = vertical_stddev
        self.camera_dist = camera_dist
        self.fov = fov
        self.perturb_mode = perturb_mode
        self.opencv_axis = opencv_axis

    def __call__(self, batch_size, resolution, **kwargs):
        """Samples points in the world space.

        Args:
            batch_size: The number of sets of points to be sampled.
        
        Returns: dict()
            pts: All points sampled in camera space. (batch_size, H, W, num_steps, 3)
            pts_z: Z values of all points sampled in camera space. (batch_size, H, W, num_steps, 3)
            ray_dirs: Directions of all rays. (batch_size, H, W, 3)
            ray_origins: Origins of all rays. (batch_size, H, W, 3)
            pitch: The pitch of the camera.
            yaw: The yaw of the camera.
        """

        for key, value in kwargs.items():
            if key in self.__dict__.keys():
                self.__dict__[key] = value

        results_cam_coord = sample_pts_in_cam_coord(batch_size=batch_size,
                                                    resolution=resolution,
                                                    num_steps=self.num_steps,
                                                    ray_start=self.ray_start,
                                                    ray_end=self.ray_end,
                                                    fov=self.fov)

        cam_pos = sample_cam_positions(num_of_pos=batch_size,
                                       camera_dist=self.camera_dist,
                                       horizontal_mean=self.horizontal_mean,
                                       horizontal_stddev=self.horizontal_stddev,
                                       vertical_mean=self.vertical_mean,
                                       vertical_stddev=self.vertical_stddev,
                                       radius=self.radius)

        pts, pts_z = perturb_points(pts=results_cam_coord['pts'],
                                    pts_z=results_cam_coord['pts_z'],
                                    ray_dirs=results_cam_coord['ray_dirs'],
                                    perturb_mode=self.perturb_mode)

        results_world_coord = trans_pts_cam2world(pts=pts,
                                                  ray_dirs=results_cam_coord['ray_dirs'],
                                                  cam_pos=cam_pos['cam_pos'])
        
        return {**results_world_coord, 'pts_z': pts_z, 'pitch': cam_pos['pitch'], 'yaw': cam_pos['yaw']}



def sample_pts_in_cam_coord(batch_size,
                            resolution,
                            num_steps,
                            ray_start,
                            ray_end,
                            fov):
    """Samples points in the camera space.

    Args:
        batch_size: The number of sets of points to be sampled.
        resolution: The resolution of the image. (W, H) or one INT number.
        num_steps: The number of points to be sampled along one ray.
    
    Returns: dict()
        pts: All points sampled in camera space. (batch_size, H, W, num_steps, 3)
        pts_z: z values of all points sampled in camera space. (batch_size, H, W, num_steps, 1)
        ray_dirs: Ray directions. (batch_size, H, W, 3)
    """

    if isinstance(resolution, tuple):
        W, H = resolution
    else:
        W = H = resolution

    device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')
    
    # Y is flipped to follow image memory layouts.
    x, y = torch.meshgrid(torch.linspace(-1, 1, W, device=device),
                            torch.linspace(1, -1, H, device=device))
    x = x.T.flatten()
    y = y.T.flatten()
    z = -torch.ones_like(x, device=device) / np.tan((2 * math.pi * fov / 360) / 2)

    # Ray directions, z values and point coordinates for each sample.
    ray_dirs = normalize_vecs(torch.stack([x, y, z], dim=-1)).to(device)
    pts_z = torch.linspace(ray_start, ray_end, num_steps, device=device).reshape(1, num_steps, 1).repeat(W*H, 1, 1)
    pts = ray_dirs.unsqueeze(1).repeat(1, num_steps, 1) * pts_z 
    
    pts = pts.unsqueeze(0).repeat(batch_size, 1, 1, 1).reshape(batch_size, H, W, num_steps, 3)
    pts_z = pts_z.unsqueeze(0).repeat(batch_size, 1, 1, 1).reshape(batch_size, H, W, num_steps, 1)
    ray_dirs = ray_dirs.unsqueeze(0).repeat(batch_size, 1, 1).reshape(batch_size, H, W, 3)

    results = {
        'pts': pts,
        'pts_z': pts_z,
        'ray_dirs': ray_dirs
    }
    
    return results


def sample_cam_positions(num_of_pos, 
                         camera_dist,
                         horizontal_mean,
                         horizontal_stddev,
                         vertical_mean,
                         vertical_stddev,
                         radius
                         ):
    """Samples camera positions according to the given camera distribution. Note that the cameras are distributed on a sphere of radius r.

    Args:
        num_of_pos: The number of cameras positions to be sampled.

    Returns:
        cam_pos: Camera positions sampled from the given distribution.
        pitch: Pitch for all sampled camera positions.

    """
    device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')

    if camera_dist == 'uniform':
        yaw = (torch.rand((num_of_pos, 1), device=device) - 0.5) * 2 * horizontal_stddev + horizontal_mean
        pitch = (torch.rand((num_of_pos, 1), device=device) - 0.5) * 2 * vertical_stddev + vertical_mean

    elif camera_dist in ['normal', 'gaussian']:
        yaw = torch.randn((num_of_pos, 1), device=device) * horizontal_stddev + horizontal_mean
        pitch = torch.randn((num_of_pos, 1), device=device) * vertical_stddev + vertical_mean

    elif camera_dist == 'hybrid':
        if random.random() < 0.5:
            yaw = (torch.rand((num_of_pos, 1), device=device) - 0.5) * 2 * horizontal_stddev * 2 + horizontal_mean
            pitch = (torch.rand((num_of_pos, 1), device=device) - 0.5) * 2 * vertical_stddev * 2 + vertical_mean
        else:
            yaw = torch.randn((num_of_pos, 1), device=device) * horizontal_stddev + horizontal_mean
            pitch = torch.randn((num_of_pos, 1), device=device) * vertical_stddev + vertical_mean

    elif camera_dist == 'truncated_gaussian':
        yaw = truncated_normal(torch.zeros((num_of_pos, 1), device=device)) * horizontal_stddev + horizontal_mean
        pitch = truncated_normal(torch.zeros((num_of_pos, 1), device=device)) * vertical_stddev + vertical_mean

    elif camera_dist == 'spherical_uniform':
        yaw = (torch.rand((num_of_pos, 1), device=device) - .5) * 2 * horizontal_stddev + horizontal_mean
        v_stddev, v_mean = vertical_stddev / math.pi, vertical_mean / math.pi
        v = ((torch.rand((num_of_pos,1), device=device) - .5) * 2 * v_stddev + v_mean)
        v = torch.clamp(v, 1e-5, 1 - 1e-5)
        pitch = torch.arccos(1 - 2 * v)

    else:
        raise ValueError(f'Invalid camera distribution: `{camera_dist}`!\n'
                        f'Types allowed: {list(_CAMERA_DIST)}.')

    pitch = torch.clamp(pitch, 1e-5, math.pi - 1e-5)

    cam_pos = torch.zeros((num_of_pos, 3), device=device)
    cam_pos[:, 0:1] = radius * torch.sin(pitch) * torch.cos(yaw)
    cam_pos[:, 2:3] = radius * torch.sin(pitch) * torch.sin(yaw)
    cam_pos[:, 1:2] = radius * torch.cos(pitch)

    results = {
        'cam_pos': cam_pos,
        'pitch': pitch,
        'yaw': yaw,
    }

    return results


def trans_pts_cam2world(pts,
                        ray_dirs,
                        cam_pos):
    """Transforms points from camera space to world space.

    Args:
        pts: Points in the camera space. (batch_size, H, W, num_steps, 3) or (batch_size, num_rays, num_steps, 3)
        ray_dirs: Directions of all rays. (batch_size, H, W, 3) or (batch_size, num_rays, 3)
        cam_pos: The position of the camera.
        
    """

    num_dims = pts.ndim
    assert num_dims in [4, 5]
    assert ray_dirs.ndim == num_dims - 1

    device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')

    if num_dims == 4:
        batch_size, num_rays, num_steps, channels = pts.shape
    else:
        batch_size, H, W, num_steps, channels = pts.shape
        num_rays = H * W
        pts = pts.reshape(batch_size, num_rays, num_steps, channels)
        ray_dirs = ray_dirs.reshape(batch_size, num_rays, 3)


    cam2world = create_cam2world_matrix(cam_pos)
    
    pts_homo = torch.ones((batch_size, num_rays, num_steps, channels+1), device=device)
    pts_homo[...,:3] = pts
    transformed_pts = torch.bmm(cam2world, pts_homo.reshape(batch_size, -1, 4).permute(0, 2, 1)).permute(
                                    0, 2, 1)[..., :3]

    transformed_ray_dirs = torch.bmm(cam2world[:, :3, :3], ray_dirs.reshape(batch_size, -1, 3).permute(0, 2, 1)).permute(
                                    0, 2, 1)

    ray_origins_homo = torch.zeros((batch_size, num_rays, 4), device=device)
    ray_origins_homo[..., 3] = 1
    transformed_ray_origins = torch.bmm(cam2world, ray_origins_homo.permute(0, 2, 1)).permute(0, 2, 1)[..., :3]

    if num_dims == 5:
        transformed_pts = transformed_pts.reshape(batch_size, H, W, num_steps, 3)
        transformed_ray_dirs = transformed_ray_dirs.reshape(batch_size, H, W, 3)
        transformed_ray_origins = transformed_ray_origins.reshape(batch_size, H, W, 3)
    
    results = {
        'pts': transformed_pts,
        'ray_dirs': transformed_ray_dirs,
        'ray_origins': transformed_ray_origins,
    }

    return results

def create_cam2world_matrix(cam_pos, opencv_axis=False):
    """Create transformation matrix for camera-to-world transformation

    Args:
        cam_pos:
        opencv_axis: Whether to get the cam2world matrix under opencv coordinate system. (Default: False)
    """

    device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')

    forward_vector = normalize_vecs(-cam_pos)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float,
                            device=device).expand_as(forward_vector)

    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector,
                                            dim=-1))

    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector,
                                        dim=-1))

    rotation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(
        forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack(
        (-left_vector, up_vector, -forward_vector), dim=-1)

    translation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(
        forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = cam_pos

    cam2world = translation_matrix @ rotation_matrix
    if opencv_axis:
        cam2world[...,1:3] = cam2world[...,1:3] * -1

    return cam2world


def perturb_points(pts, pts_z, ray_dirs, perturb_mode):
    """

    """

    device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')

    if perturb_mode == 'mid':
        # get intervals between samples
        mids = .5 * (pts_z[..., 1:, :] + pts_z[..., :-1, :])
        upper = torch.cat([mids, pts_z[..., -1:, :]], dim=-2)
        lower = torch.cat([pts_z[..., 0:1, :], mids], dim=-2)
        # uniform samples in those intervals
        t_rand = torch.rand_like(pts_z)
        pts_z = lower + (upper - lower) * t_rand 

        mids = .5 * (pts[..., 1:, :] + pts[..., :-1, :])
        upper = torch.cat([mids, pts[..., -1:, :]], dim=-2)
        lower = torch.cat([pts[..., 0:1, :], mids], dim=-2)
        # uniform samples in those intervals
        pts = lower + (upper - lower) * t_rand 
    else:
        distance_between_points = pts_z[:, :, :, 1:2, :] - pts_z[:, :, :, 0:1, :]
        offset = (torch.rand(pts_z.shape, device=device) - 0.5) * distance_between_points
        if perturb_mode == 'none':
            offset = offset * 0
        pts_z = pts_z + offset
        pts = pts + offset * ray_dirs.unsqueeze(-2)

    return pts, pts_z





if __name__ == '__main__':
    pointsampler = PointsSampling(num_steps=24,
                                  ray_start=0.8,
                                  ray_end=1.2,
                                  radius=1.0,
                                  horizontal_mean=0,
                                  horizontal_stddev=3.14,
                                  vertical_mean=0,
                                  vertical_stddev=3.14,
                                  camera_dist='gaussian',
                                  fov=30,
                                  perturb_mode='normal',
                                  opencv_axis=False)
    results = pointsampler(batch_size=8, resolution=64)
    print(results['pts'][0,0,0,0])
    print(results['pts_z'][0,0,0,0])
    print(results['ray_dirs'][0,0,0])
    print(results['ray_origins'][0,0,0])
