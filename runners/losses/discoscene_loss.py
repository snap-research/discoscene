# python3.7
"""Defines loss functions for PI-GAN training."""

import torch
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import autocast

import os
import cv2
from utils.dist_utils import ddp_sync
from .base_loss import BaseLoss
from utils.image_utils import save_image
from torch_efficient_distloss import eff_distloss, eff_distloss_native, flatten_eff_distloss


__all__ = ['DiscoSceneLoss']


class DiscoSceneLoss(BaseLoss):
    """Contains the class to compute logistic GAN loss."""

    def __init__(self, 
                 runner, 
                 d_loss_kwargs=None, 
                 g_loss_kwargs=None, 
                 use_object=False, 
                 add_scene_d=False, 
                 bbox_scale=1, 
                 add_object_head=False, 
                 object_use_pg=False, 
                 object_use_ada=False, 
                 pad_object=False, 
                 use_mask=False, 
                 dual_dist=False):
        """Initializes with models and arguments for computing losses."""
        self.d_loss_kwargs = d_loss_kwargs or dict()
        self.g_loss_kwargs = g_loss_kwargs or dict()
        self.r1_gamma = self.d_loss_kwargs.get('r1_gamma', 10.0)
        self.r2_gamma = self.d_loss_kwargs.get('r2_gamma', 0.0)
        self.latent_gamma = self.d_loss_kwargs.get('latent_gamma', 0.0)
        self.camera_gamma = self.d_loss_kwargs.get('camera_gamma', 15.0)
        self.batch_split = self.d_loss_kwargs.get('batch_split', 2)
        self.use_object = use_object
        self.pad_object = pad_object
        self.bbox_scale = bbox_scale
        self.add_object_head = add_object_head
        self.object_use_pg = object_use_pg
        self.object_use_ada = object_use_ada
        self.dual_dist = dual_dist
        self.use_mask = use_mask
        if add_scene_d:
            assert self.use_object, 'use_object must be True if add_scene_d is True'
        self.add_scene_d = add_scene_d
        self.scene_gamma = self.d_loss_kwargs.get('scene_gamma', 1.0)
        self.object_gamma = self.d_loss_kwargs.get('object_gamma', 1.0)
        self.entropy_gamma =  self.g_loss_kwargs.get('entropy_gamma', 0.0)
        self.distortion_gamma =  self.g_loss_kwargs.get('distortion_gamma', 0.0)

        if self.add_scene_d or add_object_head:
            runner.running_stats.add('Loss/Object D Real',
                                     log_name=f'object_loss_d_real',
                                     log_format='.3f',
                                     log_strategy='AVERAGE')
            runner.running_stats.add('Loss/Object D Fake',
                                     log_name=f'object_loss_d_fake',
                                     log_format='.3f',
                                     log_strategy='AVERAGE')
            runner.running_stats.add('Loss/Object G',
                                     log_name=f'object_loss_g',
                                     log_format='.3f',
                                     log_strategy='AVERAGE')
            if self.r1_gamma > 0.0:
                runner.running_stats.add('Loss/Object Real Grad Penalty',
                                         log_name='object_loss_gp_real',
                                         log_format='.2e',
                                         log_strategy='AVERAGE')

        runner.running_stats.add('Loss/D Real',
                                 log_name=f'loss_d_real',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')
        runner.running_stats.add('Loss/D Fake',
                                 log_name=f'loss_d_fake',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')
        runner.running_stats.add('Loss/G',
                                 log_name=f'loss_g',
                                 log_format='.3f',
                                 log_strategy='AVERAGE')
        if self.r1_gamma > 0.0:
            runner.running_stats.add('Loss/Real Grad Penalty',
                                     log_name='loss_gp_real',
                                     log_format='.2e',
                                     log_strategy='AVERAGE')
        if self.r2_gamma > 0.0:
            runner.running_stats.add('Loss/Fake Grad Penalty',
                                     log_name='loss_gp_fake',
                                     log_format='.2e',
                                     log_strategy='AVERAGE')

        if self.latent_gamma > 0.0 or self.camera_gamma > 0.0:
             runner.running_stats.add(f'Loss/G Fake ID Penalty', 
                                      log_format='.3f', 
                                      log_name='loss_g_id',
                                      log_strategy='AVERAGE')
             runner.running_stats.add(f'Loss/D Fake ID Penalty',
                                      log_name='loss_d_ld', 
                                      log_format='.3f', 
                                      log_strategy='AVERAGE')
        runner.running_stats.add('Loss/FG_AVG_Weights', log_name='fg_avg_weights', log_format='.4e', log_strategy='AVERAGE') 
        runner.running_stats.add('Loss/Entropy_loss', log_name='entropy_loss', log_format='.4e', log_strategy='AVERAGE') 
        runner.running_stats.add('Loss/Distortion_loss', log_name='distortion_loss', log_format='.4e', log_strategy='AVERAGE') 
        # Log loss settings.
        runner.logger.info('real gradient penalty:', indent_level=1)
        runner.logger.info(f'r1_gamma: {self.r1_gamma}', indent_level=2)
        runner.logger.info('fake gradient penalty:', indent_level=1)
        runner.logger.info(f'r2_gamma: {self.r2_gamma}', indent_level=2)
        
    @staticmethod
    def preprocess_image(images, lod=0):
        """Pre-process images to support progressive training."""
        # Downsample to the resolution of the current phase (level-of-details).
        for _ in range(int(lod)):
            images = F.avg_pool2d(
                images, kernel_size=2, stride=2, padding=0)
        # Transition from the previous phase (level-of-details) if needed.
        if lod != int(lod):
            downsampled_images = F.avg_pool2d(
                images, kernel_size=2, stride=2, padding=0)
            upsampled_images = F.interpolate(
                downsampled_images, scale_factor=2, mode='nearest')
            alpha = lod - int(lod)
            images = images * (1 - alpha) + upsampled_images * alpha
        # Upsample back to the resolution of the model.
        if int(lod) == 0:
            return images
        return F.interpolate(
            images, scale_factor=(2 ** int(lod)), mode='nearest')

    @staticmethod
    def run_G(runner, batch_size=None, sync=True, split=1, _G_kwargs=dict(), _bbox_kwargs=dict()):
        """Forwards generator."""
        G = runner.ddp_models['generator']
        G_kwargs = runner.model_kwargs_train['generator']

        # Prepare latent codes and labels.
        batch_size = batch_size or runner.batch_size
        assert batch_size % split == 0
        split_batch_size = batch_size // split
        
        latent_dim = runner.models['generator'].latent_dim
        label_dim = runner.models['generator'].label_dim
        latents = torch.randn((batch_size, *latent_dim), device=runner.device)
        labels = None
        if label_dim > 0:
            rnd_labels = torch.randint(
                0, label_dim, (batch_size,), device=runner.device)
            labels = F.one_hot(rnd_labels, num_classes=label_dim)

        with ddp_sync(G, sync=sync):
            results = {}
            for batch_idx in range(0, batch_size, split_batch_size):
                latent = latents[batch_idx:batch_idx+split_batch_size]
                label = labels[batch_idx:batch_idx+split_batch_size] if labels is not None else labels
                sub_bbox_kwargs = dict()
                for key, val in _bbox_kwargs.items():
                    sub_bbox_kwargs[key] = _bbox_kwargs[key][batch_idx:batch_idx+split_batch_size]
                result = G(latent, label, bbox_kwargs=sub_bbox_kwargs, **G_kwargs, **_G_kwargs)
                for key, val in result.items():
                    if key in results:
                        if isinstance(val, (torch.Tensor, )):
                            results[key] = torch.cat([results[key], val])
                        elif val is None:
                            results[key] = None 
                        else:
                            print(key)
                            raise NotImplementedError
                    else:
                        results[key] = val           
            return results   
        
    def postprocess(self, images):
        """Post-processes images from `torch.Tensor` to `numpy.ndarray`."""
        images = images.detach().cpu().numpy()
        images = (images + 1) * 255 / 2
        images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
        images = images.transpose(0, 2, 3, 1)
        return images

    def select(self, images, bboxes, labels=None, vis_object=False, image_raw_scale=256, object_scale=64, valid=None, masks=None, is_real=True):
        H, W = images.shape[-2:]
        scale = H/image_raw_scale[0]
        lus = (bboxes.min(dim=-2)[0]*scale)
        rds = (bboxes.max(dim=-2)[0]*scale)
        def scale_bbox(lus, rds):
            center = (lus+rds)/2
            length = (rds-lus)
            # x1, y1     x2, y2
            # TODO  consider x scale
            '''
            lus = center - self.bbox_scale*length/2
            rds = center + self.bbox_scale*length/2
            '''
            lus[:,:,0] = (center - self.bbox_scale*length/2)[:,:,0]
            rds[:,:,0] = (center + self.bbox_scale*length/2)[:,:,0]

            lus = lus.floor().int()
            rds = rds.ceil().int()
            lus = lus.clamp(0, H-1)
            rds = rds.clamp(0, W-1)
            return lus, rds
        lus, rds = scale_bbox(lus, rds)
        sub = lus - rds
        area = (sub[...,0]*sub[...,1])

        image_list = []
        bs = images.shape[0]
        numb = bboxes.shape[1]
        vis_images = []
        sub_images = []
        sub_labels = []
        if is_real and masks is not None:
            masks = F.interpolate(masks, size=64, mode='bilinear', align_corners=False)
            masks = F.interpolate(masks, size=images.shape[-1], mode='bilinear', align_corners=False)
        if valid is not None:
             valid[torch.logical_and(valid==1, area<=0)] = -1
        

        for i in range(bs):
            if vis_object:
                print(images[i:i+1].shape)
                objects = [F.interpolate(images[i:i+1], (object_scale, object_scale), mode='bilinear')]   
            r_j = 0 

            for j in range(numb): 
                image = images[i:i+1]
                label = None if labels is None else labels[i:i+1]
                is_valid = True
                if valid is not None:
                    is_valid = (valid[i, j]==1)
                if not is_valid:
                    if valid[i, r_j]!=1:
                        while valid[i, r_j]!=1:
                            r_j = (r_j + 1) % valid.shape[-1] 
                    lu, rd = lus[i,r_j], rds[i,r_j]
                    sub_image = image[:, :, lu[1]:rd[1], lu[0]:rd[0]]
                    if masks is not None: 
                        if is_real:
                            sub_mask = masks[i:i+1, r_j, lu[1]:rd[1], lu[0]:rd[0]]
                        else:
                            sub_mask = masks[i:i+1, :, lu[1]:rd[1], lu[0]:rd[0]]
                    sub_label = None if label is None else label[:, r_j]
                    r_j = (r_j + 1) % valid.shape[-1]
                else:
                    lu, rd = lus[i,j], rds[i,j]
                    sub_image = image[:, :, lu[1]:rd[1], lu[0]:rd[0]]
                    if masks is not None: 
                        if is_real:
                            sub_mask = masks[i:i+1, j, lu[1]:rd[1], lu[0]:rd[0]]
                        else:
                            sub_mask = masks[i:i+1, :, lu[1]:rd[1], lu[0]:rd[0]]
                    sub_label = None if label is None else label[:, j]

                if masks is not None:
                    # color = torch.tensor([-1, -1, 1], dtype=sub_image.dtype, device=sub_image.device).reshape(1, 3, 1, 1)
                    # sub_mask = (sub_mask[:, None]>0.5).float()
                    # sub_image = sub_mask*(sub_image*0.2 + 0.8*color ) + (1-sub_mask)*sub_image
                    if is_real: sub_mask = sub_mask[:, None]
                    sub_image = torch.cat([sub_image, sub_mask], dim=1)

                if self.pad_object:
                    a, b, h, w = sub_image.shape
                    if h > w:
                        pad_image = torch.ones((a, b, h, h), dtype=sub_image.dtype, device=sub_image.device)*-1
                        pad_image[:,:,:,(h-w)//2:(h+w)//2] = sub_image
                    elif h <= w:
                        pad_image = torch.ones((a, b, w, w), dtype=sub_image.dtype, device=sub_image.device)*-1
                        pad_image[:,:,(w-h)//2:(h+w)//2, :] = sub_image
                else:
                    pad_image = sub_image

                try:
                    sub_image = F.interpolate(pad_image, (object_scale, object_scale), mode='bilinear')
                except:
                    import ipdb;ipdb.set_trace()
                sub_images.append(sub_image)
                sub_labels.append(sub_label)
                if vis_object:
                    objects.append(sub_image)
            if vis_object:
                vis_images.append(torch.cat(objects, dim=-1))   
        sub_images = torch.cat(sub_images, dim=0)
        if labels is not None:
            sub_labels = torch.cat(sub_labels, dim=0)
        else:
            sub_labels = None
        if vis_object:
            vis_images = torch.cat(vis_images, dim=0)
            vis_images = self.postprocess(vis_images)
            vis_list = []
            for i in range(bs):
                for j in range(numb):
                    lu, rd = lus[i,j], rds[i,j]
                    vis_img = vis_images[i].copy()
                    cv2.circle(vis_img, (lu[0].item(), lu[1].item()), radius=1, color=(0,0,255), thickness=1)
                    cv2.circle(vis_img, (lu[0].item(), rd[1].item()), radius=1, color=(0,0,255), thickness=1)
                    cv2.circle(vis_img, (rd[0].item(), lu[1].item()), radius=1, color=(0,0,255), thickness=1)
                    cv2.circle(vis_img, (rd[0].item(), rd[1].item()), radius=1, color=(0,0,255), thickness=1)

                    for pts in bboxes[i, j]:
                        x, y = int(pts[0]*scale), int(pts[1]*scale)
                        cv2.circle(vis_img, (x, y), radius=1, color=(255,0,0), thickness=1)
                    vis_list.append(vis_img)
            return sub_images, vis_list
        return sub_images, sub_labels


    @staticmethod
    def run_D(runner, images, labels, sync=True, split=1, _D_kwargs=dict(), _bbox_kwargs=dict()):
        batch_size = images.shape[0]
        assert batch_size % split == 0
        split_batch_size = batch_size // split
        """Forwards discriminator."""
        D = runner.ddp_models['discriminator']
        D_kwargs = runner.model_kwargs_train['discriminator']

        with ddp_sync(D, sync=sync):
            results = {}
            for batch_idx in range(0, batch_size, split_batch_size):
                image = images[batch_idx:batch_idx+split_batch_size]
                label = labels[batch_idx:batch_idx+split_batch_size] if labels is not None else None
                sub_bbox_kwargs = {} 
                for key, val in _bbox_kwargs.items():
                    sub_bbox_kwargs[key] = _bbox_kwargs[key][batch_idx:batch_idx+split_batch_size]
                result = D(runner.augment(image, **runner.augment_kwargs), label, bbox_kwargs=sub_bbox_kwargs, **D_kwargs, **_D_kwargs)
                for key, val in result.items():
                    if key in results:
                        if isinstance(val, (torch.Tensor, )):
                            results[key] = torch.cat([results[key], val])
                        elif val is None:
                            results[key] = None 
                        else:
                            raise NotImplementedError
                    else:
                        results[key] = val           
            return results     

    @staticmethod
    def run_object_D(runner, images, labels, use_mask, sync=True, split=1, _D_kwargs=dict(),object_use_ada=False):
        batch_size = images.shape[0]
        assert batch_size % split == 0
        split_batch_size = batch_size // split
        """Forwards discriminator."""
        D = runner.ddp_models['discriminator_object']
        D_kwargs = runner.model_kwargs_train['discriminator_object']

        with ddp_sync(D, sync=sync):
            results = {}
            for batch_idx in range(0, batch_size, split_batch_size):
                image = images[batch_idx:batch_idx+split_batch_size]
                label = labels[batch_idx:batch_idx+split_batch_size] if labels is not None else None
                if object_use_ada:
                    i_sh = image.shape[1]
                    vis_mask = False 
                    if i_sh == 4 and vis_mask:
                        ori_image = image[:, :3]
                        ori_mask = image[:, 3:]

                    if i_sh == 4 and not use_mask:
                        image = image[:, :3]

                    ada_image = runner.object_augment(image, **runner.object_augment_kwargs)
                    result = D(ada_image, label, **D_kwargs, **_D_kwargs)

                    if i_sh == 4 and vis_mask:
                        ada_mask = ada_image[:,3:]
                        ada_image = ada_image[:, :3]
                    # TODO
                        result['ori_image'] = ori_image
                        result['ada_image'] = ada_image
                        result['ori_mask'] = ori_mask
                        result['ada_mask'] = ada_mask
                else:
                    result = D(image, label, **D_kwargs, **_D_kwargs)
                for key, val in result.items():
                    if key in results:
                        if isinstance(val, (torch.Tensor, )):
                            results[key] = torch.cat([results[key], val])
                        elif val is None:
                            results[key] = None
                        else:
                            raise NotImplementedError
                    else:
                        results[key] = val
            return results
            
    @staticmethod
    def compute_grad_penalty(images, scores, amp_scaler):
        """Computes gradient penalty."""
        # Scales the scores for autograd.grad's backward pass.
        # If disable amp, the scaler will always be 1.
        scores = amp_scaler.scale(scores)

        image_grad = torch.autograd.grad(
            outputs=[scores.sum()],
            inputs=[images],
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        if amp_scaler.is_enabled():
            image_grad = image_grad / amp_scaler.get_scale()

        with autocast(enabled=amp_scaler.is_enabled()):
            penalty = image_grad.square().sum((1, 2, 3))

        return penalty

     
    def d_loss(self, runner, data, sync=True, update_kwargs=dict()):
        """Computes loss for discriminator."""  
        # Update parameters for G and D
        noise_std = max(0, 1-runner.iter/5000.)
        alpha = min(1, (runner.iter-1)/self.d_loss_kwargs['fade_steps'])
        _G_kwargs = dict(noise_std=noise_std)
        _D_kwargs = dict(alpha=alpha) 
        g_bbox_kwargs = dict([(x, y)  for x, y in data.items() if 'g_' in x])
        d_bbox_kwargs = dict([(x, y)  for x, y in data.items() if 'bbox' in x and 'g_' not in x])

        real_images = data['image']
        g_image_raw_scale = data['g_image_raw_scale']

        # Train with real samples
        real_images = self.preprocess_image(
            real_images, lod=runner.lod).detach()
        real_images.requires_grad_(self.r1_gamma > 0.0)
        real_labels = data.get('label', None)

        # TODO 
        if self.dual_dist:
            down_real_images = F.interpolate(real_images, size=64, mode='bilinear', align_corners=False)
            up_real_images = F.interpolate(real_images, size=real_images.shape[-1], mode='bilinear', align_corners=False)
            real_images = torch.cat([real_images, up_real_images], dim=1)
        real_pred_results = self.run_D(runner,
                                 images=real_images,
                                 labels=real_labels,
                                 sync=False,
                                 # sync=sync,
                                 split=1,
                                 _D_kwargs=_D_kwargs,
                                 _bbox_kwargs=d_bbox_kwargs)
        real_scores = real_pred_results['score']

        with autocast(enabled=runner.enable_amp):
            d_real_loss = F.softplus(-real_scores)
            runner.running_stats.update({'Loss/D Real': d_real_loss})
            d_real_loss = runner.amp_scaler.scale(d_real_loss)
            
            # TODO DIFFAUG
            # Adjust the augmentation strength if needed.
            if hasattr(runner.augment, 'prob_tracker'):
                runner.augment.prob_tracker.update(real_scores.sign())
        
        # Train with fake samples.
        fake_results = self.run_G(
            runner, sync=False, split=self.batch_split, _G_kwargs=_G_kwargs, _bbox_kwargs=g_bbox_kwargs)
        fake_images = fake_results['image']
        fake_nerf_images = fake_results['image_raw']
        if self.use_object and not self.add_scene_d:
            fake_images = self.select(fake_images, fake_results['image_bbox'], image_raw_scale=g_image_raw_scale) 
        if self.dual_dist:
            up_fake_nerf_images = F.interpolate(fake_nerf_images, size=fake_images.shape[-1], mode='bilinear', align_corners=False) 
            fake_images = torch.cat([fake_images, up_fake_nerf_images], dim=1)
        fake_pred_results = self.run_D(runner,
                                       images=fake_images,
                                       labels=fake_results['label'],
                                       sync=sync,
                                       split=1,
                                       _D_kwargs=_D_kwargs,
                                       _bbox_kwargs=g_bbox_kwargs)

        fake_scores = fake_pred_results['score']
        with autocast(enabled=runner.enable_amp):
            d_fake_loss = F.softplus(fake_scores)
            runner.running_stats.update({'Loss/D Fake': d_fake_loss})
            d_fake_loss = runner.amp_scaler.scale(d_fake_loss)

        # Gradient penalty with real samples.
        r1_penalty = torch.zeros_like(d_real_loss)
         
        if self.r1_gamma > 0.0:
            r1_penalty = self.compute_grad_penalty(
                images=real_images,
                scores=real_scores,
                amp_scaler=runner.amp_scaler)
            runner.running_stats.update({'Loss/Real Grad Penalty': r1_penalty})
            r1_penalty = runner.amp_scaler.scale(r1_penalty)

        # Gradient penalty with fake samples.
        r2_penalty = torch.zeros_like(d_fake_loss)
        if self.r2_gamma > 0.0:
            r2_penalty = self.compute_grad_penalty(
                images=fake_results['image'],
                scores=fake_scores,
                amp_scaler=runner.amp_scaler)
            runner.running_stats.update({'Loss/Fake Grad Penalty': r2_penalty})
            r2_penalty = runner.amp_scaler.scale(r2_penalty)

        # TODO How to scale???
        # Identity penalty with fake samples.
        with autocast(enabled=runner.enable_amp):
            id_penalty = torch.zeros_like(d_fake_loss)
            if 'latent' in fake_pred_results and 'camera' in fake_pred_results:
                latent_ne = torch.numel(fake_results['latent'])
                latent_penalty = F.mse_loss(fake_pred_results['latent'].reshape(-1)[:latent_ne], fake_results['latent'].reshape(-1))
                camera_ne = torch.numel(fake_results['camera'])
                camera_penalty = F.mse_loss(fake_pred_results['camera'].reshape(-1)[:camera_ne], fake_results['camera'].reshape(-1))
                id_penalty = camera_penalty * self.camera_gamma + latent_penalty * self.latent_gamma
                runner.running_stats.update({'Loss/D Fake ID Penalty': id_penalty})
                id_penalty = runner.amp_scaler.scale(id_penalty)

        if self.add_object_head: 
            assert self.add_scene_d is False, 'add_scene_d must be False if add_object_head is True'
            real_object_scores = real_pred_results['object_score']
            with autocast(enabled=runner.enable_amp):
                d_real_object_loss = F.softplus(-real_object_scores)
                runner.running_stats.update({'Loss/Object D Real': d_real_object_loss})
                d_real_object_loss = runner.amp_scaler.scale(d_real_object_loss)

            fake_object_scores = fake_object_results['object_score']
            with autocast(enabled=runner.enable_amp):
                d_fake_object_loss = F.softplus(fake_object_scores)
                runner.running_stats.update({'Loss/Object D Fake': d_fake_object_loss})
                d_fake_object_loss = runner.amp_scaler.scale(d_fake_object_loss)

            real_images.requires_grad_(self.r1_gamma > 0.0)
            # Gradient penalty with real samples.
            r1_object_penalty = torch.zeros_like(d_real_object_loss)
            if self.r1_gamma > 0.0:
                r1_object_penalty = self.compute_grad_penalty(
                    images=real__images,
                    scores=real_object_scores,
                    amp_scaler=runner.amp_scaler)
                runner.running_stats.update({'Loss/Object Real Grad Penalty': r1_object_penalty})
                r1_object_penalty = runner.amp_scaler.scale(r1_object_penalty)
            # Gradient penalty with fake samples.
            r2_object_penalty = torch.zeros_like(d_fake_object_loss)
            if self.r2_gamma > 0.0:
                r2_object_penalty = self.compute_grad_penalty(
                    images=fake_images,
                    scores=fake_object_scores,
                    amp_scaler=runner.amp_scaler)
                runner.running_stats.update({'Loss/Object Fake Grad Penalty': r2_object_penalty})
                r2_object_penalty = runner.amp_scaler.scale(r2_object_penalty)
            r1_gamma = update_kwargs.get('r1_gamma', self.r1_gamma)
            scene_loss = (d_real_loss +
                          d_fake_loss +
                          r1_penalty * (r1_gamma * 0.5) +
                          r2_penalty * (self.r2_gamma * 0.5) +
                          id_penalty).mean()
            object_loss = (d_real_object_loss +
                           d_fake_object_loss +
                           r1_object_penalty * (r1_gamma * 0.5) +
                           r2_object_penalty * (self.r2_gamma * 0.5)).mean()
            return self.scene_gamma*scene_loss + self.object_gamma*object_loss


        if self.add_scene_d: 
            assert self.add_object_head is False, 'add_object head must be false if add_scene_d is set to be True'
            object_scale = runner.ddp_models['discriminator_object'].module.resolution
            real_object_images, real_object_labels  = self.select(real_images, data['image_bbox'], labels=d_bbox_kwargs.get('bbox_label', None), image_raw_scale=g_image_raw_scale, object_scale=object_scale, valid=d_bbox_kwargs.get('bbox_valid', None), masks=d_bbox_kwargs.get('image_bbox_mask', None), is_real=True)

            # Train with real samples
            # TODO  check with yujun whether to preprocess???
            if self.object_use_pg:
                real_object_images = self.preprocess_image(
                    real_object_images, lod=runner.lod).detach()
            real_object_images.requires_grad_(self.r1_gamma > 0.0)
            real_labels = data.get('label', None)

            # print('real_object_forward')
            real_object_results = self.run_object_D(runner,
                                     images=real_object_images,
                                     labels=real_object_labels,
                                     # sync=sync,
                                     sync=False,
                                     split=1,
                                     _D_kwargs=_D_kwargs,
                                     object_use_ada=self.object_use_ada,
                                     use_mask=self.use_mask)
            real_object_scores = real_object_results['score']

            with autocast(enabled=runner.enable_amp):
                d_real_object_loss = F.softplus(-real_object_scores)
                runner.running_stats.update({'Loss/Object D Real': d_real_object_loss})
                d_real_object_loss = runner.amp_scaler.scale(d_real_object_loss)
                
                # TODO DIFFAUG
                # Adjust the augmentation strength if needed.
                if hasattr(runner.object_augment, 'prob_tracker'):
                    runner.object_augment.prob_tracker.update(real_object_scores.sign())
            # import ipdb;ipdb.set_trace() 
            fake_object_images, fake_object_labels = self.select(fake_images, fake_results['image_bbox'], labels=g_bbox_kwargs.get('g_bbox_label', None), image_raw_scale=g_image_raw_scale, object_scale=object_scale, valid=g_bbox_kwargs.get('g_bbox_valid', None), masks=fake_results.get('norm_up_weight_map', None), is_real=False)
            # real_images, vis_images = self.select(real_images, data['image_bbox'], True, 1/4)
            # for vidx, vis_image in enumerate(vis_images):
            # print('fake_object_forward')
            fake_object_pred_results = self.run_object_D(runner,
                                           images=fake_object_images,
                                           labels=fake_object_labels,
                                           sync=sync,
                                           split=1,
                                           _D_kwargs=_D_kwargs,
                                           object_use_ada=self.object_use_ada,
                                           use_mask=self.use_mask)

            fake_object_scores = fake_object_pred_results['score']

            with autocast(enabled=runner.enable_amp):
                d_fake_object_loss = F.softplus(fake_object_scores)
                runner.running_stats.update({'Loss/Object D Fake': d_fake_object_loss})
                d_fake_object_loss = runner.amp_scaler.scale(d_fake_object_loss)

            # Gradient penalty with real samples.
            r1_object_penalty = torch.zeros_like(d_real_object_loss)
            if self.r1_gamma > 0.0:
                r1_object_penalty = self.compute_grad_penalty(
                    images=real_object_images,
                    scores=real_object_scores,
                    amp_scaler=runner.amp_scaler)
                runner.running_stats.update({'Loss/Object Real Grad Penalty': r1_object_penalty})
                r1_object_penalty = runner.amp_scaler.scale(r1_object_penalty)

            # Gradient penalty with fake samples.
            r2_object_penalty = torch.zeros_like(d_fake_object_loss)
            if self.r2_gamma > 0.0:
                r2_object_penalty = self.compute_grad_penalty(
                    images=fake_images,
                    scores=fake_object_scores,
                    amp_scaler=runner.amp_scaler)
                runner.running_stats.update({'Loss/Object Fake Grad Penalty': r2_object_penalty})
                r2_object_penalty = runner.amp_scaler.scale(r2_object_penalty)
            scene_loss = (d_real_loss +
                          d_fake_loss +
                          r1_penalty * (self.r1_gamma * 0.5) +
                          r2_penalty * (self.r2_gamma * 0.5) +
                          id_penalty).mean()
            object_loss = (d_real_object_loss +
                           d_fake_object_loss +
                           r1_object_penalty * (self.r1_gamma * 0.5) +
                           r2_object_penalty * (self.r2_gamma * 0.5)).mean()
            d_full_loss = self.object_gamma*object_loss
            if self.scene_gamma > 0:
                d_full_loss = d_full_loss + self.scene_gamma*scene_loss
            # return self.scene_gamma*scene_loss + self.object_gamma*object_loss
            return d_full_loss
        return (d_real_loss +
                d_fake_loss +
                r1_penalty * (self.r1_gamma * 0.5) +
                r2_penalty * (self.r2_gamma * 0.5) +
                id_penalty).mean()
    
    def entropy_loss(self, runner, mask, alphas, mode='object'):
        prob = alphas / (alphas.sum(dim=-2, keepdims=True)+1e-10)
        entropy = (-1*prob*torch.log2(prob+1e-10)).sum(dim=-2)
        entropy_se = entropy[mask]
        return entropy_se
        

    def distortion_loss(self, runner, mask, weights, pts_mid, intervals, mode='object'):
        m_weight = weights[mask].squeeze(-1)
        m_pts_mid = pts_mid[mask].squeeze(-1)
        m_intervals = intervals[mask].squeeze(-1)
        distortion = eff_distloss(m_weight, m_pts_mid, m_intervals)
        return distortion 
         
    def g_loss(self, runner, _data, sync=True, update_kwargs=dict()):
        """Computes loss for generator."""
        # Update parameters for G and D
        noise_std = max(0, 1-runner.iter/5000.)
        alpha = min(1, (runner.iter-1)/self.d_loss_kwargs['fade_steps'])
        _G_kwargs = dict(noise_std=noise_std)
        _D_kwargs = dict(alpha=alpha) 
        _bbox_kwargs = dict([(x, y)  for x, y in _data.items() if 'g_' in x])
        
        topk_percent = 1
        if 'topk_interval' in self.g_loss_kwargs and 'topk_v' in self.g_loss_kwargs: 
            topk_percent = max(0.99**(runner.iter/self.g_loss_kwargs['topk_interval']),
                               self.g_loss_kwargs['topk_v'])
        topk_num = int(topk_percent * runner.batch_size) 

        fake_results = self.run_G(runner, sync=sync, split=self.batch_split, _G_kwargs=_G_kwargs, _bbox_kwargs=_bbox_kwargs)
        fake_images = fake_results['image']
        fake_nerf_images = fake_results['image_raw']
        g_image_raw_scale = _data['g_image_raw_scale']
        if self.use_object and not self.add_scene_d:
            fake_images = self.select(fake_images, fake_results['image_bbox'], image_raw_scale=g_image_raw_scale)
        if self.dual_dist:
            up_fake_nerf_images = F.interpolate(fake_nerf_images, size=fake_images.shape[-1], mode='bilinear', align_corners=False) 
            fake_images = torch.cat([fake_images, up_fake_nerf_images], dim=1)
        fake_pred_results = self.run_D(runner,
                                 images=fake_images,
                                 labels=fake_results['label'],
                                 sync=False,
                                 split=1,
                                 _D_kwargs=_D_kwargs,
                                 _bbox_kwargs=_bbox_kwargs)
        runner.running_stats.update({'Loss/FG_AVG_Weights': fake_results['avg_weights'].mean()})

        if self.entropy_gamma > 0:
            entropy_loss = self.entropy_loss(runner=runner,
                                             mask=fake_results['ray_mask'],
                                             alphas=fake_results['alphas'],
                                             mode='object',
                                             )
            runner.running_stats.update({'Loss/Entropy_loss': entropy_loss.mean()})

        if self.distortion_gamma > 0:
            distortion_loss = self.distortion_loss(runner=runner,
                                             mask=fake_results['ray_mask'],
                                             weights=fake_results['weights'],
                                             pts_mid=fake_results['pts_mid'],
                                             intervals=fake_results['intervals'],
                                             mode='object',
                                             )
            runner.running_stats.update({'Loss/Distortion_loss': distortion_loss.mean()})
        
        with autocast(enabled=runner.enable_amp):
            fake_scores = torch.topk(fake_pred_results['score'], topk_num, dim=0).values
            g_loss = F.softplus(-fake_scores)
            runner.running_stats.update({'Loss/G': g_loss})
            g_loss = runner.amp_scaler.scale(g_loss)

        # TODO How to scale???
        # Identity penalty with fake samples.
        with autocast(enabled=runner.enable_amp):
            id_penalty = torch.zeros_like(g_loss)
            if 'latent' in fake_pred_results and 'camera' in fake_pred_results:
                latent_ne = torch.numel(fake_results['latent'])
                latent_penalty = F.mse_loss(fake_pred_results['latent'].reshape(-1)[:latent_ne], fake_results['latent'].reshape(-1))
                camera_ne = torch.numel(fake_results['camera'])
                camera_penalty = F.mse_loss(fake_pred_results['camera'].reshape(-1)[:camera_ne], fake_results['camera'].reshape(-1))
                # latent_penalty = F.mse_loss(fake_pred_results['latent'], fake_results['latent'].reshape(fake_pred_results['latent'].shape))
                # camera_penalty = F.mse_loss(fake_pred_results['camera'], fake_results['camera'])
                id_penalty = camera_penalty * self.camera_gamma + latent_penalty * self.latent_gamma
                runner.running_stats.update({'Loss/G Fake ID Penalty': id_penalty})
                id_penalty = runner.amp_scaler.scale(id_penalty)

        if self.add_object_head:
            assert self.add_scene_d is False, 'add_scene_d must be False if add_object_head is True' 
            num_bbox = fake_results['image_bbox'].shape[1]
            with autocast(enabled=runner.enable_amp):
                topk_object_num = int(topk_percent * runner.batch_size * num_bbox)
                fake_object_scores = torch.topk(fake_pred_results['object_score'], topk_object_num, dim=0).values
                g_object_loss = F.softplus(-fake_object_scores)
                runner.running_stats.update({'Loss/Object G': g_object_loss})
                g_object_loss = runner.amp_scaler.scale(g_object_loss)
            return self.scene_gamma*((g_loss + id_penalty).mean()) + self.object_gamma*(g_object_loss.mean())

        if self.add_scene_d:
            assert self.add_object_head is False, 'add_object head must be false if add_scene_d is set to be True'
            object_scale = runner.ddp_models['discriminator_object'].module.resolution
            fake_object_images, fake_object_labels = self.select(fake_images, fake_results['image_bbox'], labels=_bbox_kwargs.get('g_bbox_label', None), image_raw_scale=g_image_raw_scale, object_scale=object_scale, valid=_bbox_kwargs.get('g_bbox_valid', None), masks=fake_results.get('norm_up_weight_map', None), is_real=False)
            num_bbox = fake_results['image_bbox'].shape[1]
            fake_object_pred_results = self.run_object_D(runner,
                                           images=fake_object_images,
                                           labels=fake_object_labels,
                                           sync=False,
                                           split=1,
                                           _D_kwargs=_D_kwargs,
                                           object_use_ada=self.object_use_ada,
                                           use_mask=self.use_mask)
            with autocast(enabled=runner.enable_amp):
                topk_object_num = int(topk_percent * runner.batch_size * num_bbox) 
                fake_object_scores = torch.topk(fake_object_pred_results['score'], topk_object_num, dim=0).values
                g_object_loss = F.softplus(-fake_object_scores)
                runner.running_stats.update({'Loss/Object G': g_object_loss})
                g_object_loss = runner.amp_scaler.scale(g_object_loss)    

            g_full_loss = self.object_gamma*(g_object_loss.mean())
            if self.scene_gamma > 0:
                g_full_loss = g_full_loss + self.scene_gamma*((g_loss + id_penalty).mean())
            if self.entropy_gamma:
                g_full_loss = g_full_loss + self.entropy_gamma* (entropy_loss.mean())
            if self.distortion_gamma:
                g_full_loss = g_full_loss + self.distortion_gamma * (distortion_loss.mean())
            return g_full_loss
            # if self.entropy_gamma > 0:
            #     return self.scene_gamma*((g_loss + id_penalty).mean()) + self.object_gamma*(g_object_loss.mean()) + self.entropy_gamma*(entropy_loss.mean())
            # else:
            #     return self.scene_gamma*((g_loss + id_penalty).mean()) + self.object_gamma*(g_object_loss.mean()) 
        return (g_loss + id_penalty).mean()
