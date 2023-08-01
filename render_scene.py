# python3.7
"""A simple tool to synthesize images with pre-trained models."""

import os
import math
import click
import random

import skvideo.io
from tqdm import tqdm
import torch
import numpy as np
import cv2
import copy

from configs import build_config, CONFIG_POOL
from models import build_model
import imageio

from datasets import build_dataset
from utils.parsing_utils import parse_bool, DictAction
from utils.visualizers import HtmlVisualizer
from utils.image_utils import save_image, load_image, resize_image
from utils.misc import gather_data


def postprocess(images):
    """Post-processes images from `torch.Tensor` to `numpy.ndarray`."""
    images = images.detach().cpu().numpy()
    images = (images + 1) * 255 / 2
    images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
    images = images.transpose(0, 2, 3, 1)
    return images

def preprocess(images):
    """Pre-process images from `numpy array` to `torch tensor`"""
    images = torch.from_numpy(images.astype(np.float32)).cuda() 
    images = images*2.0/255.0 - 1.0
    images = images.permute(0, 3, 1, 2) 
    return images

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz*0, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

def rotation_from_axis(theta, axis):
    rotdir = np.array(axis) * theta
    rotmat, _ = cv2.Rodrigues(rotdir)
    return rotmat

def delete_object(_bbox_kwargs, code, tidx):
    bbox_kwargs = copy.deepcopy(_bbox_kwargs)
    bbox_num = bbox_kwargs['g_bbox'].shape[1]
    idxs = [x for x in range(bbox_num) if x != tidx]
    code = code[:, idxs+[-1]]
    keys = ['g_bbox', 'g_bbox_tran', 'g_bbox_rot', 'g_bbox_scale', 'g_bbox_valid', 'bbox_s', 'bbox_c']
    for key in keys:
        bbox_kwargs[key] = bbox_kwargs[key][:, idxs]
    return bbox_kwargs, code

def add_object(_bbox_kwargs, code, tidx, t, dataset_type):
    bbox_kwargs = copy.deepcopy(_bbox_kwargs)

    bbox_num = bbox_kwargs['g_bbox'].shape[1]
    idxs = [x for x in range(bbox_num) ] + [tidx]
    new_code = code[:, idxs+[-1]]
    keys = ['g_bbox', 'g_bbox_tran', 'g_bbox_rot', 'g_bbox_scale', 'g_bbox_valid', 'bbox_s', 'bbox_c']
    for key in keys:
        bbox_kwargs[key] = bbox_kwargs[key][:, idxs]
    g_bbox = bbox_kwargs['g_bbox']
    trans = bbox_kwargs['g_bbox_tran']

    if dataset_type == 'clevr':
        directions = torch.tensor([3.5, 4, 0]).to(trans.device).to(trans.dtype)
    elif dataset_type == '3dfront':
        directions = torch.tensor([0, -4, 0]).to(trans.device).to(trans.dtype)
    else:
        directions = torch.tensor([12, 0, 0]).to(trans.device).to(trans.dtype)
    if t >= 0.5: t = 1 - t
    directions = directions.reshape(1, 3)

    for i in [tidx]:
        trans[:, i] = trans[:, i] +  t*directions.reshape(trans[:, 0].shape)
        g_bbox[:, i] = g_bbox[:, i] + t*directions[:, None]

    bbox_kwargs['g_bbox'] = g_bbox
    bbox_kwargs['g_bbox_tran'] = trans
    return bbox_kwargs, new_code 


def move_object(_bbox_kwargs, t, dataset_type):
    bbox_kwargs = copy.deepcopy(_bbox_kwargs)
    g_bbox = bbox_kwargs['g_bbox']
    trans = bbox_kwargs['g_bbox_tran']

    if dataset_type == 'clevr':
        directions = torch.tensor([3.5, 4, 0]).to(trans.device).to(trans.dtype)
    elif dataset_type == '3dfront':
        directions = torch.tensor([3.0, 0.0, 0]).to(trans.device).to(trans.dtype)
    else:
        directions = 2*torch.tensor([0, 0, 4]).to(trans.device).to(trans.dtype)
    directions = directions.reshape(1, 3)
    if t >= 0.5: t = 1-t

    for idx in range(trans.shape[1]):
        trans[:, idx] = trans[:, idx] +  t*directions.reshape(trans[:, 0].shape)
        g_bbox[:, idx] = g_bbox[:, idx] + t*directions[:, None]

    bbox_kwargs['g_bbox'] = g_bbox
    bbox_kwargs['g_bbox_tran'] = trans
    return bbox_kwargs

def rotate_object(_bbox_kwargs, t, dataset_type):
    bbox_kwargs = copy.deepcopy(_bbox_kwargs)    
    cano_bbox = bbox_kwargs['g_cano_bbox']
    bs = cano_bbox.shape[0]
    directions = -2*math.pi
    theta = directions*t 
    if dataset_type == 'clevr':
        rot = rotation_from_axis(theta, [0, 0, 1])
    elif dataset_type == 'waymo':
        rot = rotation_from_axis(theta, [0, 1, 0])
    elif dataset_type == '3dfront':
        rot = rotation_from_axis(theta, [0, 0, 1])
    trans = bbox_kwargs['g_bbox_tran']
    scales = bbox_kwargs['g_bbox_scale']
    
    rot = torch.tensor(rot, device=trans.device, dtype=trans.dtype)
    rot = rot[None].repeat(bs, 1, 1)

    align_angle = False 
    for idx in range(cano_bbox.shape[1]):
        if align_angle:
            _rot = rot
        else:
            _rot = rot @ bbox_kwargs['g_bbox_rot'][:, idx]
        bbox = cano_bbox[:, idx]
        if dataset_type == 'clevr':
            scale = scales[..., idx].reshape(bs, 1, -1)
        else:
            scale = scales[:, idx].reshape(bs, 1, scales.shape[-1])
        tran = trans[:, idx].reshape(bs, 1, 3)
        pts = (_rot @ bbox.permute(0, 2, 1)).permute(0, 2, 1)
        pts = pts * scale  + tran
        
        bbox_kwargs['g_bbox'][:,idx] = pts
        bbox_kwargs['g_bbox_rot'][:, idx] = _rot
    return bbox_kwargs

def move_camera(RT, t):
    import copy
    RT = RT.astype(np.float32)
    R = RT[:, :3]
    T = RT[:, 3:]

    new_RT = copy.deepcopy(RT)
    directions = np.array([0, 0, 3.]).astype(T.dtype)
    if t >= 0.5: t = 1-t
    directions = directions.reshape(T.shape)
    T = T + t*directions
    new_RT[:,3:] = T

    new_RT = torch.tensor(new_RT)
    return new_RT

def rotate_camera(RT, t):
    def normalize_vecs(vectors): return vectors / (torch.norm(vectors, dim=-1, keepdim=True))
    import copy
    RT = torch.tensor(RT)[:, [0, 2, 1, 3]]
    R = RT[:, :3].T
    T = -RT[:, :3].T @ RT[:,3:]
    
    norm_T = normalize_vecs(T.reshape(-1))
    yaw = torch.arctan(norm_T[2]/norm_T[0])
    pitch = torch.arccos(norm_T[1])

    directions = 2*math.pi
    yaw = yaw + math.pi + directions*t

    r = torch.norm(T)
    y = r*torch.cos(pitch)
    x = r*torch.sin(pitch)*torch.cos(yaw)
    z = r*torch.sin(pitch)*torch.sin(yaw)
    cam_pos = torch.stack([x, y, z]).reshape(-1)
    
    forward_vector = normalize_vecs(-cam_pos)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float,
                                        device=R.device).reshape(-1).expand_as(forward_vector)
    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector,
                                                        dim=-1))

    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector,
                                                        dim=-1))
    rotate = torch.stack(
                    (left_vector, -up_vector, forward_vector), dim=-1)
    new_R =  rotate.T
    new_T =  new_R @ -cam_pos.reshape(3,1)
    new_RT = copy.deepcopy(RT)
    new_RT[:,:3] = new_R
    new_RT[:,3:] = new_T
    new_RT = new_RT[:, [0, 2, 1, 3]]

    return new_RT

@click.group(name='Render Script',
             help='Render image, video',
             context_settings={'show_default': True, 'max_content_width': 180})
@click.option('--checkpoint', type=str,
              help='Path to the checkpoint to load.')
@click.option('--work_dir', type=str, default='work_dirs/synthesis',
              help='Directory to save the results. If not specified, '
                   'the results will be saved to '
                   '`work_dirs/synthesis/` by default.')
@click.option('--num', type=int, default=10,
              help='Number of samples to synthesize.')
@click.option('--batch_size', type=int, default=1,
              help='Batch size.')
@click.option('--step', type=int, default=70,
              help='Render video steps')
@click.option('--seed', type=int, default=0,
              help='Seed for sampling.')
@click.option('--row_num', type=int, default=5,
              help='Number of videos per row')
@click.option('--render_type', type=click.Choice(['rotate_object', 'move_object', 'rotate_camera', 'move_camera', 'delete_object', 'add_object']), default='rotate_object',
              help='Choose the type of the render results')
@click.option('--generate_html', type=parse_bool, default=True,
              help='Whether to generate html.')
@click.option('--generate_gif', type=parse_bool, default=False,
              help='Whether to generate gif.')
@click.option('--dataset_type', type=click.Choice(['clevr', '3dfront', 'waymo']), default='clevr',
              help='specify the dataset type')
@click.option('--code_path', type=str, default=None,
              help='code path')
@click.option('--ssaa', type=int, default=None,
              help='the upsampling ratio for super-sample anti-aliasing')
def command_group(checkpoint, work_dir, num, batch_size, step, seed, row_num, render_type, generate_gif, generate_html, dataset_type, code_path, ssaa):  # pylint: disable=unused-argument
    """Defines a command group for rendering script.

    This function is mainly inherited train.py.
    """


@command_group.result_callback()
@click.pass_context
def main(ctx, kwargs, 
         checkpoint, 
         work_dir,
         num, 
         batch_size, 
         step, 
         seed, 
         row_num, 
         render_type, 
         generate_gif, 
         generate_html, 
         dataset_type, 
         code_path,
         ssaa):

    config = build_config(ctx.invoked_subcommand, kwargs).get_config()
    test_loader = build_dataset(
            for_training=True,
            batch_size=batch_size,
            dataset_kwargs=config.data.val,
            dataset_only=True)

    # CUDNN settings.
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # load checkpoint
    state = torch.load(checkpoint, map_location='cpu')
    print('finish load!')
    G_args = state['model_kwargs_init']['generator_smooth']
    G = build_model(**G_args)
    G.load_state_dict(state['models']['generator_smooth'], strict=True)
    G.eval().cuda()
    G_kwargs= dict(noise_mode='const',
                   fused_modulate=False,
                   impl='cuda',
                   fp16_res=None)

    os.makedirs(work_dir, exist_ok=True)
    job_name = f'{ctx.invoked_subcommand}_{num}'

    # set results path 
    print(f'Synthesizing {num} videos...')
    videos_path = os.path.join(work_dir, render_type, 'videos')
    gifs_path = os.path.join(work_dir, render_type, 'gifs')
    os.makedirs(videos_path, exist_ok=True)
    os.makedirs(gifs_path, exist_ok=True)
    if generate_html:
        html = HtmlVisualizer(num_rows=num, num_cols=step)


    num_bbox = test_loader.num_bbox
    G.num_bbox = num_bbox
    if code_path is not None:
        code = np.load(code_path)
        code = code[:num]
        num = len(code)
        code = torch.tensor(code).cuda()
    else:
        code = torch.randn(200, num_bbox+1, G.z_dim).cuda()
    ps_kwargs = {}
    indices = list(range(num))
    all_frames = [[] for i in range(step)]
    for batch_idx in tqdm(range(0, num, batch_size), leave=False):
        sub_indices = indices[batch_idx:batch_idx + batch_size]
        sub_code = code[sub_indices]
        _sub_code = code[sub_indices]
        sub_frames = [[] for i in sub_indices]
        cidx = (np.random.randint(len(test_loader)))
        _bbox_kwargs = gather_data([test_loader.get_bbox(cidx) for i in range(len(sub_code))], device=sub_code.device)

        bbox_centers = _bbox_kwargs['g_bbox'].reshape(len(sub_code), num_bbox, 8, 3).mean(dim=-2)
        bbox_scales = _bbox_kwargs['g_bbox_scale'].reshape(len(sub_code), num_bbox, -1)*2
        bbox_mask = ((_bbox_kwargs['g_bbox_valid']+1)/2)[..., None]
        bbox_centers = bbox_mask * bbox_centers
        bbox_scales = bbox_mask * bbox_scales
        _bbox_kwargs['bbox_s'] = bbox_scales
        _bbox_kwargs['bbox_c'] = bbox_centers


        with torch.no_grad():
            for tidx, t in tqdm(enumerate(np.linspace(0, 1, step)), leave=False):
                G_kwargs['trunc_psi'] = 0.7 
                G_kwargs['trunc_layers'] = 8 
                ps_kwargs['num_steps'] = 18 
                ps_kwargs['bg_num_steps'] = 12 
                ps_kwargs['test_resolution'] = 64 
                if ssaa:
                    ps_kwargs['test_resolution'] = 64*ssaa 

                if dataset_type == 'waymo':
                    ps_kwargs['perturb_mode'] = 'none'

                if render_type == 'rotate_object':
                    bbox_kwargs = rotate_object(_bbox_kwargs, t, dataset_type)
                elif render_type == 'move_object':
                    bbox_kwargs = move_object(_bbox_kwargs, t, dataset_type)
                    G.num_bbox = sub_code.shape[1]-1
                elif render_type == 'rotate_camera':
                    bbox_kwargs = _bbox_kwargs
                    RT = _bbox_kwargs['g_bbox_RT'].float()
                    RT = RT[0]
                    RT = RT.detach().cpu().numpy()
                    RT = rotate_camera(RT, t)
                    ps_kwargs['cam_pos'] = RT
                elif render_type == 'move_camera':
                    bbox_kwargs = _bbox_kwargs
                    RT = _bbox_kwargs['g_bbox_RT'].float()
                    RT = RT[0]
                    RT = RT.detach().cpu().numpy()
                    RT = move_camera(RT, t)
                    ps_kwargs['cam_pos'] = RT
                elif render_type == 'delete_object':
                    bbox_kwargs, sub_code = delete_object(_bbox_kwargs, _sub_code, 0)
                    G.num_bbox = sub_code.shape[1]-1
                elif render_type == 'add_object':
                    bbox_kwargs, sub_code = add_object(_bbox_kwargs, _sub_code, 0, t, dataset_type)
                    G.num_bbox = sub_code.shape[1]-1
                else:
                    raise NotImplementedError

                G_results = G(sub_code,  foreground_only=False, background_only=False, ps_kwargs=ps_kwargs, bbox_kwargs=bbox_kwargs, )
                images = G_results['image']
                ray_mask = G_results['ray_mask']
                # images = G_results['weights_map']
                # images = G_results['image_raw']
                images = postprocess(images)
                for sidx, (sub_frame, image) in enumerate(zip(sub_frames, images)):
                    image = np.ascontiguousarray(image, dtype=np.uint8).copy()
                    bboxes = bbox_kwargs['g_bbox'][sidx].detach().cpu().numpy()
                    sub_frame.append(image)
                    if generate_html:
                        html.set_cell(sub_indices[sidx], tidx, image=image, text=f'image:{tidx:05d}, step:{t:03f}')
            for sub_idx, sub_frame in zip(sub_indices, sub_frames): 
                writer = skvideo.io.FFmpegWriter(f'{videos_path}/{sub_idx:06d}.mp4', outputdict={'-pix_fmt': 'yuv420p', '-crf': '21'})
                for fidx, f in enumerate(sub_frame):
                    writer.writeFrame(f)
                    all_frames[fidx].append(f)
                writer.close()
                if generate_gif:
                    os.makedirs(os.path.join(work_dir, 'gifs'), exist_ok=True)
                    imageio.mimsave(f'{gifs_path}/{sub_idx:06d}.gif', sub_frame, duration=1/21)

    if generate_html:
        html.save(os.path.join(work_dir, f'{render_type}/{job_name}_{render_type}_images.html'))

    all_num = num
    if num // row_num == 0:
        row_num = num
    elif num % row_num != 0:
        all_num = int(row_num * (num // row_num))
    all_cat_frames = []
    for x in all_frames:
        row_list = []
        for all_sidx in range(int(all_num//row_num)):
            row_list.append(np.concatenate(x[(all_sidx*row_num):(all_sidx+1)*row_num], axis=1))
        all_cat_frames.append(np.concatenate(row_list, axis=0))
    all_writer = skvideo.io.FFmpegWriter(f'{videos_path}/full_{num}.mp4', outputdict={'-pix_fmt': 'yuv420p', '-crf': '21'})
    for all_frame in all_cat_frames:
        all_writer.writeFrame(all_frame)
    all_writer.close()

    print(f'Finish synthesizing {num} videos.')

if __name__ == '__main__':
    # Append all available commands (from `configs/`) into the command group.
    for cfg in CONFIG_POOL:
        command_group.add_command(cfg.get_command())
    # Run by interacting with command line.
    command_group()  # pylint: disable=no-value-for-parameter
