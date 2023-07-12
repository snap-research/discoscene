#4 python3.7
"""Contains the function to sample the points in 3D space."""

import torch
import numpy as np
import math
import random

from models.clib import aabb_ray_intersect
from .utils import normalize_vecs
from .utils import truncated_normal

__all__ = ['PointsSampling', 'sample_pts_in_cam_coord', 'sample_cam_positions', 'trans_pts_cam2world', 'create_cam2world_matrix', 'perturb_points']
_CAMERA_DIST = ['uniform', 'normal', 'gaussian', 'hybrid', 'truncated_gaussian', 'spherial_uniform']

class PointsBboxSampling(object):
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
                 cam_path=None,
                 voxel_size=2,
                 transform_type='clevr',
                 bg_num_steps=None,
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
        self.voxel_size = voxel_size
        self.voxel_size = 2.2
        self.transform_type = transform_type
        self.aspect_ratio = 1
        self.bg_num_steps = bg_num_steps
        if self.transform_type == 'nuscenes':
            self.aspect_ratio = 1
            self.aspect_ratio = 9/16
        if self.transform_type == 'waymo':
            self.aspect_ratio = 2/3
        if self.transform_type == '3dfront':
            self.aspect_ratio = 1 

    def __call__(self, batch_size, resolution, bbox_kwargs,  bg_resolution, **kwargs):
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
        # TODO bbox_transforms: ROT, Trans, Scale
        for key, value in kwargs.items():
            if key in self.__dict__.keys():
                self.__dict__[key] = value
        # 1. sample ray_dirs and ray_origins
        if self.transform_type == 'clevr':
            results_cam_coord = sample_rays_in_cam_coord(batch_size=batch_size,
                                                  resolution=resolution,
                                                  num_steps=self.num_steps,
                                                  ray_start=self.ray_start,
                                                  ray_end=self.ray_end,
                                                  fov=self.fov)
            if bg_resolution is not None:
                bg_results_cam_coord = sample_rays_in_cam_coord(batch_size=batch_size,
                                                      resolution=bg_resolution,
                                                      num_steps=self.num_steps,
                                                      ray_start=self.ray_start,
                                                      ray_end=self.ray_end,
                                                      fov=self.fov,
                                                      prefix='bg_')
        elif self.transform_type in ['nuscenes', 'waymo', '3dfront']:
            assert 'g_bbox_K' in bbox_kwargs
            results_cam_coord = sample_rays_in_cam_coord_by_K(batch_size=batch_size,
                                                  resolution=resolution,
                                                  num_steps=self.num_steps,
                                                  ray_start=self.ray_start,
                                                  ray_end=self.ray_end,
                                                  K=bbox_kwargs['g_bbox_K'],
                                                  image_scale=bbox_kwargs['g_image_raw_scale'])
            if bg_resolution is not None:
                bg_results_cam_coord = sample_rays_in_cam_coord_by_K(batch_size=batch_size,
                                                  resolution=bg_resolution,
                                                  num_steps=self.num_steps,
                                                  ray_start=self.ray_start,
                                                  ray_end=self.ray_end,
                                                  K=bbox_kwargs['g_bbox_K'],
                                                  image_scale=bbox_kwargs['g_image_raw_scale'],
                                                  prefix='bg_')

        assert 'g_bbox_K' in bbox_kwargs
        assert 'g_bbox_RT' in bbox_kwargs
        cam_pos = kwargs.get('cam_pos', bbox_kwargs['g_bbox_RT'])
        if cam_pos.ndim == 2:
            cam_pos = cam_pos.unsqueeze(0)
        cam_pos = cam_pos.contiguous()
        cam_pos_ = sample_cam_positions(num_of_pos=batch_size,
                                       camera_dist=self.camera_dist,
                                       horizontal_mean=self.horizontal_mean,
                                       horizontal_stddev=self.horizontal_stddev,
                                       vertical_mean=self.vertical_mean,
                                       vertical_stddev=self.vertical_stddev,
                                       radius=self.radius)

        results_world_coord = trans_pts_cam2world(ray_oris=results_cam_coord['ray_oris'],
                                                  ray_dirs=results_cam_coord['ray_dirs'],
                                                  cam_pos=cam_pos,
                                                  )
        if bg_resolution is not None:
            bg_results_world_coord = trans_pts_cam2world(ray_oris=bg_results_cam_coord['bg_ray_oris'],
                                                  ray_dirs=bg_results_cam_coord['bg_ray_dirs'],
                                                  cam_pos=cam_pos,
                                                  prefix='bg_'
                                                  )
            results_world_coord.update(bg_results_world_coord)
        # 3. calculate near and far depth for bboxes 
        # TODO consider scale matrix
        # print('self.voxel_size', self.voxel_size)
        # print('self.transform_type', self.transform_type)
        intersect_results = intersect_ray_bbox(ray_dirs=results_world_coord['ray_dirs'],
                                               ray_oris=results_world_coord['ray_oris'],
                                               bbox_kwargs=bbox_kwargs,
                                               inverse=True,
                                               voxel_size=self.voxel_size,
                                               transform_type=self.transform_type,
                                               )
        # 4. sample points between near and far in world coordinate
        pts_results = sample_pts_by_depth(ray_dirs=results_world_coord['ray_dirs'],
                                          ray_oris=results_world_coord['ray_oris'],
                                          pts_idx=intersect_results['pts_idx'],    
                                          min_depth=intersect_results['min_depth'],
                                          max_depth=intersect_results['max_depth'],
                                          perturb_mode=self.perturb_mode,
                                          ray_start=self.ray_start,
                                          ray_end=self.ray_end,
                                          aspect_ratio=self.aspect_ratio,
                                          num_steps=self.num_steps,
                                          bg_ray_dirs=results_world_coord.get('bg_ray_dirs',None),
                                          bg_ray_oris=results_world_coord.get('bg_ray_oris',None),
                                          bg_num_steps=self.bg_num_steps,
                                          transform_bboxes=intersect_results['transform_bboxes'])
        # 5. transform forground into bbox system we keep background points
        fg_pts_object = trans_pts_world2object(pts=pts_results['fg_pts'],
                                             bbox_kwargs=bbox_kwargs,
                                             inverse=True,
                                             transform_type=self.transform_type,
                                             transform_bboxes=intersect_results['transform_bboxes'])
        pts_results.update(fg_pts_object=fg_pts_object,
                           ray_dirs_world=results_world_coord.get('bg_ray_dirs', results_world_coord['ray_dirs']),
                           ray_dirs_object=intersect_results['transform_ray_dirs'],
                           ray_oris_object=intersect_results['transform_ray_oris']) 
        pts_results = {**pts_results, 'pitch': cam_pos_['pitch'], 'yaw': cam_pos_['yaw']}
        if 'lock_view' in bbox_kwargs:
            pts_results.update(ray_dirs_lock=intersect_results['lock_ray_dirs'])
        return pts_results
'''
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
'''
def trans_pts_world2object(pts, bbox_kwargs, inverse=False, transform_type='clevr', transform_bboxes=None):
    for k in bbox_kwargs:
        bbox_kwargs[k] = bbox_kwargs[k].to(pts.device)
    bboxes, cano_bboxes, scale_mat, trans_mat, rot_mat = bbox_kwargs['g_bbox'], bbox_kwargs['g_cano_bbox'], bbox_kwargs['g_bbox_scale'], bbox_kwargs['g_bbox_tran'], bbox_kwargs['g_bbox_rot']

    pts_dim = pts.ndim
    if pts_dim == 6:
        bs, N, H, W, NS, C = pts.shape
        pts = pts.reshape(bs, N, -1, NS, C) 
    bs, N, L, NS, C = pts.shape
    if inverse:
        rot_mat = rot_mat.reshape(bs, N, 1, 3, 3)
        scale_mat = scale_mat.reshape(bs, N, 1, 1, -1)
        trans_mat = trans_mat.reshape(bs, N, 1, 1, 3)
        if transform_type in ['clevr', '3dfront']:
            transform_pts = (rot_mat.permute(0, 1, 2, 4, 3) @ ((pts - trans_mat)/(scale_mat+1e-8)).permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)
        elif transform_type in ['nuscenes', 'waymo', ]:
            transform_pts = ((rot_mat.permute(0, 1, 2, 4, 3) @ (pts - trans_mat).permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3))/(scale_mat+1e-8)
        else:
            raise NotImplementedError

    else: raise NotImplementedError
    if pts_dim == 6:
        transform_pts = transform_pts.reshape(bs, N, H, W, NS, C)
    return transform_pts

def sample_pts_by_depth(ray_dirs, ray_oris, pts_idx, min_depth, max_depth, num_steps, perturb_mode, ray_start, ray_end, aspect_ratio=1, res=None, bg_ray_dirs=None, bg_ray_oris=None, bg_num_steps=None, transform_bboxes=None):
    """
    Args:
       ray_dirs:  [bs, L, 3]
       pts_idx:   [bs, N, L]
       min_depth: [bs, N, L]
       max_depth: [bs, N, L]
    """
    bs, N, H, W = pts_idx.shape
    device = pts_idx.device
    
    ray_dirs = ray_dirs.reshape(bs, H*W, 3)
    ray_oris = ray_oris.reshape(bs, H*W, 3)
    max_depth = max_depth.reshape(bs, N, H*W)
    min_depth = min_depth.reshape(bs, N, H*W)

    ray_mask = (pts_idx!=-1) # for every bbox 
    fg_mask = (pts_idx.sum(dim=1)>=0) # foreground
    bg_mask = (pts_idx.sum(dim=1)<0)# background
    
    # fg_points 
    depth_range = torch.linspace(0, 1, num_steps, device=device).reshape(1, 1, 1, num_steps, 1)
    fg_pts_depth = min_depth[..., None, None] + (max_depth - min_depth)[...,None, None] * depth_range # [bs, N, L, ns, 1]
    fg_pts = ray_oris[:, None, :, None, :] + ray_dirs[:, None, :, None, :] * fg_pts_depth # [bs, 1, L, ns, 3] + [bs, N, L, 1, 1] = [bs, N, L, ns, 3]
    # perturb points
    fg_pts, fg_pts_depth = perturb_points(pts=fg_pts,
                                    pts_z=fg_pts_depth,
                                    ray_dirs=ray_dirs[:, None],
                                    perturb_mode=perturb_mode)
                                    # TODO
                                    # perturb_mode='none')

    if bg_ray_oris is None and bg_ray_dirs is None:
        bg_ray_oris = ray_oris
        bg_ray_dirs = ray_dirs  
        bH, bW = H, W 
    else:
        # bg_points
        bH, bW = bg_ray_dirs.shape[1:3]    
        bg_ray_dirs = bg_ray_dirs.reshape(bs, bH*bW, 3)
        bg_ray_oris = bg_ray_oris.reshape(bs, bH*bW, 3)

    if bg_num_steps is not None:
        bg_depth_range = torch.linspace(0, 1, bg_num_steps, device=device).reshape(1, 1, 1, bg_num_steps, 1)
    else:
        bg_depth_range = depth_range
        bg_num_steps = num_steps
    
    bg_pts_depth = ray_start + ray_end * bg_depth_range # [bs, 1, L, ns, 1]
    bg_pts_depth = bg_pts_depth.repeat(bs, 1, bH*bW, 1, 1)
    bg_pts = bg_ray_oris[:, None, :, None, :] + bg_ray_dirs[:, None, :, None, :] * bg_pts_depth
    # perturb bg points
    # TODO revert it 
    bg_pts, bg_pts_depth = perturb_points(pts=bg_pts,
                                    pts_z=bg_pts_depth,
                                    ray_dirs=bg_ray_dirs[:, None],
                                    perturb_mode=perturb_mode)
                                    # perturb_mode='none')

    fg_pts = fg_pts.reshape(bs, N, H, W, num_steps, 3)
    fg_pts_depth = fg_pts_depth.reshape(bs, N, H, W, num_steps, 1)
    ray_mask = ray_mask.reshape(bs, N, H, W)
    # for i in range(4): mask = ray_mask[0, i].reshape(64*2, 64*2, 1).repeat(1, 1, 3).detach().cpu().numpy().astype(np.uint8); import cv2;cv2.imwrite(f'mask_{i}.png', mask*255)


    bg_mask = bg_mask.reshape(bs, H, W)
    fg_mask = fg_mask.reshape(bs, H, W)
    bg_pts = bg_pts.reshape(bs, 1, bH, bW, bg_num_steps, 3)
    bg_pts_depth = bg_pts_depth.reshape(bs, 1, bH, bW, bg_num_steps, 1)
        
    results = {'fg_pts': fg_pts,
               'fg_pts_depth': fg_pts_depth,
               'ray_mask': ray_mask,
               'fg_mask': fg_mask,
               'bg_mask': bg_mask,
               'bg_pts': bg_pts,
               'bg_pts_depth': bg_pts_depth}  
    if aspect_ratio != 1:
        bg_ray_mask = (torch.zeros((bs, 1, bH, bW)) == 1)
        bH_ = int(aspect_ratio * bH)
        bW_ = bW
        bg_ray_mask[:,:,(bW_-bH_)//2:(bW_+bH_)//2,] = True
        results.update(bg_ray_mask=bg_ray_mask)
    return results


def intersect_ray_bbox(ray_dirs, ray_oris, bbox_kwargs, voxel_size=1, inverse=False, transform_type='clevr'):
    """
    Args:
        ray_dirs: [bs, H, W: [bs, H, W, 3]
        bboxes:   [bs, N, 6]
        bbox_transforms: [bs, N, 4, 4]
    """
    bs, H, W, C = ray_dirs.shape
    for k in bbox_kwargs:
        bbox_kwargs[k] = bbox_kwargs[k].to(ray_dirs.dtype)
    bboxes, cano_bboxes, scale_mat, trans_mat, rot_mat = bbox_kwargs['g_bbox'], bbox_kwargs['g_cano_bbox'], bbox_kwargs['g_bbox_scale'], bbox_kwargs['g_bbox_tran'], bbox_kwargs['g_bbox_rot']  
    bbox_valid = bbox_kwargs.get('g_bbox_valid', None)
    N = bboxes.shape[1]

    # Repeat batch dimension
    ray_dirs = ray_dirs.reshape(bs, 1, H*W, C).repeat(1, N, 1, 1).reshape(bs*N, H*W, C)
    ray_oris = ray_oris.reshape(bs, 1, H*W, C).repeat(1, N, 1, 1).reshape(bs*N, H*W, C)

    # transform ray_dir and ray_ori into bbox system
    if inverse:
        # import ipdb;ipdb.set_trace()
        scale_mat = scale_mat.reshape(bs*N, 1, -1)
        trans_mat = trans_mat.reshape(bs*N, 1, 3)
        rot_mat = rot_mat.reshape(bs*N, 3, 3)
        bboxes = bboxes.reshape(bs*N, bboxes.shape[2], 3)
        if transform_type in ['clevr', '3dfront']:
            transform_ray_dirs = (rot_mat.permute(0, 2, 1) @ (ray_dirs/(scale_mat+1e-8)).permute(0,2,1)).permute(0,2,1)
            transform_ray_oris = (rot_mat.permute(0, 2, 1) @ ((ray_oris - trans_mat)/(scale_mat+1e-8)).permute(0,2,1)).permute(0,2,1)
            transform_bboxes = (rot_mat.permute(0, 2, 1) @ ((bboxes - trans_mat)/(scale_mat+1e-8)).permute(0,2,1)).permute(0,2,1)
        elif transform_type in ['nuscenes', 'waymo', ]:
            transform_ray_dirs = (rot_mat.permute(0, 2, 1) @ (ray_dirs).permute(0,2,1)).permute(0,2,1)/(scale_mat+1e-8)
            transform_ray_oris = (rot_mat.permute(0, 2, 1) @ (ray_oris - trans_mat).permute(0,2,1)).permute(0,2,1)/(scale_mat+1e-8)
            transform_bboxes = (rot_mat.permute(0, 2, 1) @ (bboxes - trans_mat).permute(0,2,1)).permute(0,2,1)/(scale_mat+1e-8)
        else:
            raise NotImplementedError
        transform_bboxes_center = transform_bboxes.mean(dim=1, keepdim=True) 
        if 'lock_view' in bbox_kwargs:
            lock_rot = bbox_kwargs['lock_view']
            lock_rot_mat = lock_rot.reshape(bs*N, 3, 3)
            lock_ray_dirs = (lock_rot_mat.permute(0, 2, 1) @ (ray_dirs/(scale_mat+1e-8)).permute(0,2,1)).permute(0,2,1)
    else: raise NotImplementedError
    # bbox_mat = bbox_transforms.reshape(bs*N, 4, 4)
    # ray_dirs_homo = torch.ones(bs*N, H*W, 4)
    # ray_dirs_homo[:, :3] = ray_dirs
    # transform_ray_dirs = (bbox_mat @ ray_dirs_homo.permute(0, 2, 1)).permute(0, 2, 1)[..., :3]

    # ray_oris_homo = torch.ones(bs*N, H*W, 4)
    # ray_oris_homo[:, :3] = ray_oris
    # transform_ray_oris = (bbox_mat @ ray_oris_homo.permute(0, 2, 1)).permute(0, 2, 1)[..., :3]

    # # reshape bbox?
    # bboxes = bboxes.reshape(bs*N, 2, 3)
    # # align bouding box to axis
    # bboxes_homo = torch.ones(bs*N, 2, 4)
    # bboxes_homo[:, :, :3] = bboxes 
    # transform_bboxes = (bbox_mat @ bboxes_homo.permute(0, 2, 1)).permute(0, 2, 1)[..., :3]
    # transform_bboxes_center = transform_bboxes.mean(dim=1, keepdim=True)
    # # TODO support cuda code with different voxel size
    # bboxes_size = (transform_bboxes[:,0] - transform_bboxes[:,1]).abs()

    # aabb in object system
    pts_idx, min_depth, max_depth = aabb_ray_intersect(
                        voxel_size, 1, transform_bboxes_center, transform_ray_oris, transform_ray_dirs)
    # pts_idx: [bs*N, L, 1]
    # min_depth: [bs*N, L, 1]
    pts_idx = pts_idx.reshape(bs, N, H, W)

    
    min_depth = min_depth.reshape(bs, N, H, W)
    max_depth = max_depth.reshape(bs, N, H, W)
    if bbox_valid is not None:
        bbox_valid = bbox_valid.reshape(bbox_valid.shape + (1, 1))
        bbox_valid[bbox_valid == -1] = 0
        pts_idx = bbox_valid * pts_idx + (bbox_valid-1)

        min_depth = bbox_valid * min_depth
        max_depth = bbox_valid * max_depth

    num_pixels = (pts_idx!=-1).sum()
    # print('num_pixels', num_pixels.item())

    transform_ray_dirs = transform_ray_dirs.reshape(bs, N, H, W, 3) 
    transform_ray_oris = transform_ray_oris.reshape(bs, N, H, W, 3)
    results = {'pts_idx': pts_idx,
               'min_depth': min_depth,
               'max_depth': max_depth,
               'transform_ray_dirs': transform_ray_dirs,
               'transform_ray_oris': transform_ray_oris,
               'transform_bboxes': transform_bboxes}
    if 'lock_view' in bbox_kwargs: 
        lock_ray_dirs = lock_ray_dirs.reshape(bs, N, H, W, 3)
        results.update(lock_ray_dirs=lock_ray_dirs)
    return results

def sample_rays_in_cam_coord(batch_size,
                             resolution,
                             num_steps,
                             ray_start,
                             ray_end,
                             fov,
                             prefix=''):
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
                            torch.linspace(-1, 1, H, device=device))
    x = x.T.flatten()
    y = y.T.flatten()
    z = torch.ones_like(x, device=device) / np.tan((2 * math.pi * fov / 360) / 2)

    # Ray directions, z values and point coordinates for each sample.
    ray_dirs = normalize_vecs(torch.stack([x, y, z], dim=-1)).to(device)
    ray_dirs = ray_dirs.unsqueeze(0).repeat(batch_size, 1, 1).reshape(batch_size, H, W, 3)
    ray_oris = torch.zeros_like(ray_dirs)
    # results = {
    #     'ray_dirs': ray_dirs,
    #     'ray_oris': ray_oris,
    # }
    results = {
        f'{prefix}ray_dirs': ray_dirs,
        f'{prefix}ray_oris': ray_oris,
    }
    
    return results


def sample_rays_in_cam_coord_by_K(batch_size,
                             resolution,
                             num_steps,
                             ray_start,
                             ray_end,
                             K,
                             image_scale,
                             prefix=''):

    if isinstance(resolution, tuple):
        W, H = resolution
    else:
        W = H = resolution

    device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')

    # Y is flipped to follow image memory layouts.
    if not isinstance(image_scale[0], (list, tuple)):
        image_scale = (image_scale[0].item(), image_scale[0].item())
    ori_X = image_scale[1]
    x_range = torch.linspace(0, 1, W, device=device)*ori_X
    ori_Y = image_scale[0]
    y_range = torch.linspace(0, 1, H, device=device)*ori_Y
    # x, y = torch.meshgrid(torch.linspace(-1, 1, W, device=device),
    #                         torch.linspace(-1, 1, H, device=device))
    x, y = torch.meshgrid(x_range, y_range)
    x = x.T.flatten()
    y = y.T.flatten()
    z = torch.ones_like(x)
    xyz = torch.stack([x, y, z], dim=-1) # H, W, 3 

    inv_K = K.inverse().to(device).to(x_range.dtype) # 8 3*3
    xyz = xyz.reshape(1, H*W, 3).repeat(batch_size, 1, 1)
    xyz = (inv_K @ xyz.permute(0, 2, 1)).permute(0, 2, 1)
    xyz = xyz.reshape(batch_size, H, W, 3)
    # K @ xyz
    # Ray directions, z values and point coordinates for each sample.
    ray_dirs = normalize_vecs(xyz)
    # ray_dirs = ray_dirs.unsqueeze(0).repeat(batch_size, 1, 1).reshape(batch_size, H, W, 3)
    ray_oris = torch.zeros_like(ray_dirs)
    results = {
        f'{prefix}ray_dirs': ray_dirs,
        f'{prefix}ray_oris': ray_oris,
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


def trans_pts_cam2world(ray_oris,
                        ray_dirs,
                        cam_pos,
                        prefix=''):
    """Transforms points from camera space to world space.

    Args:
        pts: Points in the camera space. (batch_size, H, W, num_steps, 3) or (batch_size, num_rays, num_steps, 3)
        ray_dirs: Directions of all rays. (batch_size, H, W, 3) or (batch_size, num_rays, 3)
        cam_pos: The position of the camera.
        
    """

    num_dims = ray_oris.ndim
    assert num_dims in [3, 4]

    device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')

    if num_dims == 3:
        batch_size, num_rays, channels = ray_dirs.shape
    else:
        batch_size, H, W, channels = ray_dirs.shape
        num_rays = H * W
        ray_dirs = ray_dirs.reshape(batch_size, num_rays, 3)
        ray_oris = ray_oris.reshape(batch_size, num_rays, 3)
    # if num_dims == 4:
    #     batch_size, num_rays, num_steps, channels = pts.shape
    # else:
    #     batch_size, H, W, num_steps, channels = pts.shape
    #     num_rays = H * W
    #     pts = pts.reshape(batch_size, num_rays, num_steps, channels)
    #     ray_dirs = ray_dirs.reshape(batch_size, num_rays, 3)

    if isinstance(cam_pos, dict):
        cam2world = create_cam2world_matrix(cam_pos)
    else:
        if cam_pos.ndim == 2:
            cam2world = torch.eye(4).to(device)
            cam2world[:3] = cam_pos.to(device)
            cam2world = cam2world.unsqueeze(0).repeat(batch_size, 1, 1).inverse()
        elif cam_pos.ndim == 3:
            cam2world = torch.eye(4).to(device)
            cam2world = cam2world.unsqueeze(0).repeat(batch_size, 1, 1)
            cam2world[:, :3] = cam_pos.to(device)
            cam2world = cam2world.inverse()
    # pts_homo[...,:3] = pts
    # transformed_pts = torch.bmm(cam2world, pts_homo.reshape(batch_size, -1, 4).permute(0, 2, 1)).permute(
    #                                 0, 2, 1)[..., :3]
    transformed_ray_dirs = torch.bmm(cam2world[:, :3, :3], ray_dirs.reshape(batch_size, -1, 3).permute(0, 2, 1)).permute(
                                    0, 2, 1)

    ray_origins_homo = torch.ones((batch_size, num_rays, 4), device=device)
    ray_origins_homo[..., :3] = ray_oris

    transformed_ray_origins = torch.bmm(cam2world, ray_origins_homo.permute(0, 2, 1)).permute(0, 2, 1)[..., :3]

    if num_dims == 4:
        transformed_ray_dirs = transformed_ray_dirs.reshape(batch_size, H, W, 3)
        transformed_ray_origins = transformed_ray_origins.reshape(batch_size, H, W, 3)
    
    results = {
        f'{prefix}ray_dirs': transformed_ray_dirs,
        f'{prefix}ray_oris': transformed_ray_origins,
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
    # results = {'cam2world': cam2wolrd,
    #            'translate_mat': translation_matrix,
    #            'rotation_mat': rotation_matrix}
    return cam2world 


def perturb_points(pts, pts_z, ray_dirs, perturb_mode):
    """
    Args:
        pts:   [bs, N, L, ns, 3]
        pts_z: [bs, N, L, ns, 1]
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
        #TODO check depth variation is constant along 
        distance_between_points = pts_z[:, :, :, 1:2, :] - pts_z[:, :, :, 0:1, :]
        offset = (torch.rand(pts_z.shape, device=device) - 0.5) * distance_between_points
        if perturb_mode == 'none':
            offset = offset * 0
        pts_z = pts_z + offset
        pts = pts + offset * ray_dirs.unsqueeze(-2)

    return pts, pts_z

if __name__ == '__main__':
    print(torch.cuda.current_device())
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
    bs = 2
    results = pointsampler(batch_size=bs, resolution=64)
    ray_dirs = results['ray_dirs']
    ray_oris = results['ray_oris'] 
    translation = torch.tensor([0,0,1]).to(ray_oris.device).reshape(1, 1, 1, 3) + ray_oris
    ray_oris = ray_oris + translation
    from models.clib import aabb_ray_intersect
    voxel_size = 1
    bboxes = torch.tensor([[[0,0,0], [0,0,-1]],[[0,0,-2], [0,0,-3]]]).to(ray_oris.device)
    # bboxes = torch.tensor([[0,0,0], ]).to(ray_oris.device)
    bbox_num = bboxes.shape[1]
    # bboxes = bboxes.unsqueeze(0).repeat(bs, 1, 1)
    bboxes = bboxes.reshape(-1, 1, 3)
    # ray_dirs = ray_dirs.reshape(bs,-1,3).repeat_interleave(bbox_num, dim=0)
    # ray_oris = ray_oris.reshape(bs,-1,3).repeat_interleave(bbox_num, dim=0)
    ray_dirs = ray_dirs.reshape(bs,-1,3)
    ray_dirs[1, :, -1] = ray_dirs[1, :, -1]*-1
    ray_oris = ray_oris.reshape(bs,-1,3)
    ray_dirs = ray_dirs.unsqueeze(1).repeat(1, bbox_num,1,1).reshape(bs*bbox_num,-1, 3 )
    ray_oris = ray_oris.unsqueeze(1).repeat(1, bbox_num,1,1).reshape(bs*bbox_num,-1, 3 )
    pts_idx, min_depth, max_depth = aabb_ray_intersect(
                            voxel_size, 3, bboxes, ray_oris, ray_dirs)
    print(pts_idx.shape)
