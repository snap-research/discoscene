# python3.7
"""Configuration for training StyleGAN2."""
import math
from .base_config import BaseConfig

__all__ = ['DiscoSceneConfig']

# RUNNER = 'StyleGAN2BboxRunner'
RUNNER = 'DiscoSceneRunner'
DATASET = 'ImageBboxDataset'
DISCRIMINATOR = 'StyleGAN2Discriminator'
GENERATOR = 'DiscoSceneGenerator'
LOSS = 'DiscoSceneLoss'

PI = math.pi


class DiscoSceneConfig(BaseConfig):
    """Defines the configuration for training StyleGAN2."""

    name = 'discoscene'
    hint = 'Train a DiscoScene model.'
    info = '''
To train a DiscoScene model, the recommended settings are as follows:

\b
- batch_size: 8 
- val_batch_size: 16 
- data_repeat: 200 
- total_img: 25_000_000 
'''

    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.config.runner_type = RUNNER

    @classmethod
    def get_options(cls):
        options = super().get_options()

        options['Data transformation settings'].extend([
            cls.command_option(
                '--resolution', type=cls.int_type, default=256,
                help='Resolution of the training images.'),
            cls.command_option(
                '--image_channels', type=cls.int_type, default=3,
                help='Number of channels of the training images.'),
            cls.command_option(
                '--min_val', type=cls.float_type, default=-1.0,
                help='Minimum pixel value of the training images.'),
            cls.command_option(
                '--max_val', type=cls.float_type, default=1.0,
                help='Maximum pixel value of the training images.')
        ])

        options['Network settings'].extend([
            cls.command_option(
                '--g_init_res', type=cls.int_type, default=4,
                help='The initial resolution to start convolution with in '
                     'generator.'),
            cls.command_option(
                '--latent_dim', type=cls.int_type, default=512,
                help='The dimension of the latent space.'),
            cls.command_option(
                '--label_dim', type=cls.int_type, default=0,
                help='Number of classes in conditioning training. Set to `0` '
                     'to disable conditional training.'),
            cls.command_option(
                '--d_fmaps_factor', type=cls.float_type, default=1.0,
                help='A factor to control the number of feature maps of '
                     'discriminator, which will be `factor * 32768`.'),
            cls.command_option(
                '--d_mbstd_groups', type=cls.int_type, default=4,
                help='Number of groups for MiniBatchSTD layer of '
                     'discriminator.'),
            cls.command_option(
                '--g_fmaps_factor', type=cls.float_type, default=1.0,
                help='A factor to control the number of feature maps of '
                     'generator, which will be `factor * 32768`.'),
            cls.command_option(
                '--triplane_fmaps_factor', type=cls.float_type, default=1.0,
                help='A factor to control the number of feature maps of '
                     'generator, which will be `factor * 32768`.'),
            cls.command_option(
                '--g_num_mappings', type=cls.int_type, default=8,
                help='Number of mapping layers of generator.'),
            cls.command_option(
                '--g_architecture', type=str, default='skip',
                help='Architecture type of generator.'),
            cls.command_option(
                '--d_architecture', type=str, default='resnet',
                help='Architecture type of discriminator.'),
            cls.command_option(
                '--impl', type=str, default='cuda',
                help='Control the implementation of some neural operations.'),
            cls.command_option(
                '--num_fp16_res', type=cls.int_type, default=0,
                help='Number of (highest) resolutions that use `float16` '
                     'precision for training, which speeds up the training yet '
                     'barely affects the performance. The official '
                     'StyleGAN-ADA uses 4 by default.')
        ])

        options['Training settings'].extend([
            cls.command_option(
                '--d_lr', type=cls.float_type, default=0.002,
                help='The learning rate of discriminator.'),
            cls.command_option(
                '--d_beta_1', type=cls.float_type, default=0.0,
                help='The Adam hyper-parameter `beta_1` for discriminator '
                     'optimizer.'),
            cls.command_option(
                '--d_beta_2', type=cls.float_type, default=0.99,
                help='The Adam hyper-parameter `beta_2` for discriminator '
                     'optimizer.'),
            cls.command_option(
                '--g_lr', type=cls.float_type, default=0.002,
                help='The learning rate of generator.'),
            cls.command_option(
                '--g_beta_1', type=cls.float_type, default=0.0,
                help='The Adam hyper-parameter `beta_1` for generator '
                     'optimizer.'),
            cls.command_option(
                '--g_beta_2', type=cls.float_type, default=0.99,
                help='The Adam hyper-parameter `beta_2` for generator '
                     'optimizer.'),
            cls.command_option(
                '--w_moving_decay', type=cls.float_type, default=0.995,
                help='Decay factor for updating `w_avg`.'),
            cls.command_option(
                '--sync_w_avg', type=cls.bool_type, default=False,
                help='Synchronizing the update of `w_avg` across replicas.'),
            cls.command_option(
                '--style_mixing_prob', type=cls.float_type, default=0.9,
                help='Probability to perform style mixing as a training '
                     'regularization.'),
            cls.command_option(
                '--r1_gamma', type=cls.float_type, default=10.0,
                help='Factor to control the strength of gradient penalty.'),
            cls.command_option(
                '--r1_interval', type=cls.int_type, default=16,
                help='Interval (in iterations) to perform gradient penalty.'),
            cls.command_option(
                '--pl_batch_shrink', type=cls.int_type, default=2,
                help='Factor to reduce the batch size for perceptual path '
                     'length regularization.'),
            cls.command_option(
                '--pl_weight', type=cls.float_type, default=2.0,
                help='Factor to control the strength of perceptual path length '
                     'regularization.'),
            cls.command_option(
                '--pl_decay', type=cls.float_type, default=0.01,
                help='Decay factor for perceptual path length regularization.'),
            cls.command_option(
                '--pl_interval', type=cls.int_type, default=4,
                help='Interval (in iterations) to perform perceptual path '
                     'length regularization.'),
            cls.command_option(
                '--g_ema_img', type=cls.int_type, default=10_000,
                help='Factor for updating the smoothed generator, which is '
                     'particularly used for inference.'),
            cls.command_option(
                '--g_ema_rampup', type=cls.float_type, default=0.0,
                help='Rampup factor for updating the smoothed generator, which '
                     'is particularly used for inference. Set as `0` to '
                     'disable warming up.'),
            cls.command_option(
                '--use_ada', type=cls.bool_type, default=False,
                help='Whether to use adaptive augmentation pipeline.'),
            cls.command_option(
                '--background_only', type=cls.bool_type, default=False,
                help='Whether to use background to overfit dataset'),
            cls.command_option(
                '--static_background', type=cls.bool_type, default=False,
                help='Whether to use static background'),
            cls.command_option(
                '--background_path', type=str, default='data/clevr2_ann/bk.png',
                help='background image path'),
            cls.command_option(
                '--use_object', type=cls.bool_type, default=False,
                help='Whether to use adaptive augmentation pipeline.'),
            cls.command_option(
                '--pad_object', type=cls.bool_type, default=False,
                help='Whether to use adaptive augmentation pipeline.'),
            cls.command_option(
                '--use_bbox_2d', type=cls.bool_type, default=False,
                help='Whether to use adaptive augmentation pipeline.'),
            cls.command_option(
                '--use_hs', type=cls.bool_type, default=False,
                help='Whether to use hierachical sampling .'),
            cls.command_option(
                '--add_scene_d', type=cls.bool_type, default=False,
                help='Whether to use scene_discriminator.'),
            cls.command_option(
                '--add_object_head', type=cls.bool_type, default=False,
                help='Whether to use object_head'),
            cls.command_option(
                '--fg_use_dirs', type=cls.bool_type, default=True,
                help='Whether to use view directions for fg generator.'),
            cls.command_option(
                '--bg_use_dirs', type=cls.bool_type, default=True,
                help='Whether to use view directions for fg generator.'),
            cls.command_option(
                '--cam_path', type=str, default='data/clevr2_ann/K_RT.npy',
                help='camera path'),
            cls.command_option(
                '--bbox_scale', type=cls.float_type, default=1.0,
                help='bbox_scale'),
            cls.command_option(
                '--scene_gamma', type=cls.float_type, default=1.0,
                help='scene gamma'),
            cls.command_option(
                '--object_gamma', type=cls.float_type, default=1.0,
                help='object gamma'),
            cls.command_option(
                '--entropy_gamma', type=cls.float_type, default=0.0,
                help='entropy gamma'),
            cls.command_option(
                '--distortion_gamma', type=cls.float_type, default=0.0,
                help='distortion gamma'),
            cls.command_option(
                '--use_sr', type=cls.bool_type, default=False,
                help='Whether to use super resolution'),
            cls.command_option(
                '--nerf_feature_dim', type=cls.int_type, default=3,
                help='nerf feature dim'),
            cls.command_option(
                '--nerf_resolution', type=cls.int_type, default=64,
                help='Nerf resolution'),
            cls.command_option(
                '--bg_nerf_resolution', type=cls.int_type, default=None,
                help='Nerf resolution'),
            cls.command_option(
                '--use_stylegan2_d', type=cls.bool_type, default=False,
                help='Whether to use stylegan2d'),
            cls.command_option(
                '--d_add_coords', type=cls.bool_type, default=False,
                help='Whether to use coord conv'),
            cls.command_option(
                '--use_pg', type=cls.bool_type, default=False,
                help='Whether to use progressive training'),
            cls.command_option(
                '--object_resolution', type=cls.int_type, default=64,
                help='object_d resolution'),
            cls.command_option(
                '--use_small_lr', type=cls.bool_type, default=False,
                help='whether to use small lr'),
            cls.command_option(
                '--object_use_pg', type=cls.bool_type, default=False,
                help='whether to use small lr'),
            cls.command_option(
                '--debug', type=cls.bool_type, default=False,
                help='whether to use small lr'),
            cls.command_option(
                '--enable_beta_mult', type=cls.bool_type, default=False,
                help='whether to use small lr'),
            cls.command_option(
                '--reset_optimizer', type=cls.bool_type, default=True,
                help='whether to reset_optimizer'),
            cls.command_option(
                '--num_bbox', type=cls.int_type, default=2,
                help='bbox number'),
            cls.command_option(
                '--pg_imgs', type=cls.int_type, default=2500_000,
                help='bbox number'),
            cls.command_option(
                '--ada_milestone', type=cls.int_type, default=None,
                help='bbox number'),
            cls.command_option(
                '--top_v', type=cls.float_type, default=0.6,
                help='bbox number'),
            cls.command_option(
                '--num_steps', type=cls.int_type, default=12,
                help='bbox number'),
            cls.command_option(
                '--bg_num_steps', type=cls.int_type, default=12,
                help='bbox number'),
            cls.command_option(
                '--ps_type', type=str, default='clevr',
                help='bbox number'),
            cls.command_option(
                '--voxel_size', type=cls.float_type, default=2.0,
                help='bbox number'),
            cls.command_option(
                '--object_use_ada', type=cls.bool_type, default=False,
                help='whether to use object_ada'),
            cls.command_option(
                '--objectada_w_spatial', type=cls.bool_type, default=True,
                help='whether to use object_ada'),
            cls.command_option(
                '--ada_type', type=str, default='adaptive',
                help='runner type'),
            cls.command_option(
                '--object_ada_target_p', type=cls.float_type, default=None,
                help='ada target p'),
            cls.command_option(
                '--ada_target_p', type=cls.float_type, default=0.6,
                help='ada target p'),
            cls.command_option(
                '--object_ada_type', type=str, default=None,
                help='runner type'),
            cls.command_option(
                '--object_ada_milestone', type=cls.int_type, default=None,
                help='bbox number'),
            cls.command_option(
                '--n_freq', type=cls.int_type, default=10,
                help='runner type'),
            cls.command_option(
                '--use_mask', type=cls.bool_type, default=False,
                help='runner type'),
            cls.command_option(
                '--ada_w_color', type=cls.bool_type, default=True,
                help='runner type'),
            cls.command_option(
                '--norm_feature', type=cls.bool_type, default=True,
                help='runner type'),
            cls.command_option(
                '--sample_rotation', type=cls.bool_type, default=False,
                help='runner type'),
            cls.command_option(
                '--condition_sample', type=cls.bool_type, default=False,
                help='runner type'),
            cls.command_option(
                '--bg_condition_bbox', type=cls.bool_type, default=False,
                help='runner type'),
            cls.command_option(
                '--fg_condition_bbox', type=cls.bool_type, default=False,
                help='runner type'),
            cls.command_option(
                '--noise_type', type=str, default='spatial',
                help='noise type'),
            cls.command_option(
                '--kernel_size', type=cls.int_type, default=3,
                help='noise type'),
            cls.command_option(
                '--dual_dist', type=cls.bool_type, default=False,
                help='noise type'),
            cls.command_option(
                '--use_bbox_label', type=cls.bool_type, default=False,
                help='noise type'),
            cls.command_option(
                '--use_triplane', type=cls.bool_type, default=False,
                help='noise type'),
            cls.command_option(
                '--enable_flip', type=cls.bool_type, default=False,
                help='noise type'),
            cls.command_option(
                '--d_optimize_prob', type=cls.float_type, default=1.0,
                help='noise type'),
        ])

        return options

    @classmethod
    def get_recommended_options(cls):
        recommended_opts = super().get_recommended_options()
        recommended_opts.extend([
            'resolution', 'num_fp16_res', 'latent_dim', 'label_dim', 'd_lr',
            'g_lr', 'd_fmaps_factor', 'd_mbstd_groups', 'g_fmaps_factor',
            'g_num_mappings', 'g_ema_img', 'style_mixing_prob', 'use_ada',
            'r1_gamma', 'r1_interval', 'pl_weight', 'pl_interval'
        ])
        return recommended_opts

    def parse_options(self):
        super().parse_options()
        debug = self.args.pop('debug')
        self.config.interactive = self.config.interactive or debug
        resolution = self.args.pop('resolution')
        image_channels = self.args.pop('image_channels')
        min_val = self.args.pop('min_val')
        max_val = self.args.pop('max_val')
        use_object = self.args.pop('use_object')
        use_hs = self.args.pop('use_hs')
        use_sr = self.args.pop('use_sr')
        add_scene_d = self.args.pop('add_scene_d')
        add_object_head = self.args.pop('add_object_head')
        cam_path = self.args.pop('cam_path')
        if cam_path.lower() == 'none': cam_path = None
        bbox_scale = self.args.pop('bbox_scale')
        nerf_feature_dim = self.args.pop('nerf_feature_dim')
        nerf_resolution = self.args.pop('nerf_resolution')
        bg_nerf_resolution = self.args.pop('bg_nerf_resolution')
        use_stylegan2_d = self.args.pop('use_stylegan2_d')
        d_add_coords = self.args.pop('d_add_coords')
        object_resolution = self.args.pop('object_resolution')
        use_pg = self.args.pop('use_pg')
        use_small_lr = self.args.pop('use_small_lr')
        object_use_pg = self.args.pop('object_use_pg')
        enable_beta_mult = self.args.pop('enable_beta_mult')
        reset_optimizer = self.args.pop('reset_optimizer')
        scene_gamma = self.args.pop('scene_gamma')
        object_gamma = self.args.pop('object_gamma')
        entropy_gamma = self.args.pop('entropy_gamma')
        distortion_gamma = self.args.pop('distortion_gamma')
        num_bbox = self.args.pop('num_bbox')
        pg_imgs = self.args.pop('pg_imgs')
        top_v = self.args.pop('top_v')
        ps_type = self.args.pop('ps_type')
        voxel_size = self.args.pop('voxel_size')
        object_use_ada = self.args.pop('object_use_ada')
        object_ada_target_p = self.args.pop('object_ada_target_p')
        ada_target_p = self.args.pop('ada_target_p')
        object_ada_type = self.args.pop('object_ada_type')
        object_ada_milestone = self.args.pop('object_ada_milestone')

        objectada_w_spatial = self.args.pop('objectada_w_spatial')
        ada_type = self.args.pop('ada_type')
        ada_milestone = self.args.pop('ada_milestone')
        ada_w_color= self.args.pop('ada_w_color')
        norm_feature = self.args.pop('norm_feature')
        num_steps = self.args.pop('num_steps')
        bg_num_steps = self.args.pop('bg_num_steps')
        pad_object = self.args.pop('pad_object')
        use_bbox_2d = self.args.pop('use_bbox_2d')
        sample_rotation = self.args.pop('sample_rotation')
        condition_sample = self.args.pop('condition_sample')
        bg_condition_bbox = self.args.pop('bg_condition_bbox')
        fg_condition_bbox = self.args.pop('fg_condition_bbox')
        n_freq = self.args.pop('n_freq')
        use_mask = self.args.pop('use_mask')
        noise_type = self.args.pop('noise_type')
        kernel_size = self.args.pop('kernel_size')
        dual_dist = self.args.pop('dual_dist')
        use_bbox_label = self.args.pop('use_bbox_label')
        use_triplane = self.args.pop('use_triplane')
        label_dim = self.args.pop('label_dim')
        d_optimize_prob = self.args.pop('d_optimize_prob')
        enable_flip = self.args.pop('enable_flip')
        if object_ada_target_p is None:
            object_ada_target_p = ada_target_p
        if object_ada_type is None:
            object_ada_type = ada_type
        if object_ada_milestone is None:
            object_ada_milestone = ada_milestone
        if not use_bbox_label:
            label_dim = 0


        batch_size = self.config.batch_size
        self.config.object_use_pg = object_use_pg
        if use_stylegan2_d: assert self.config.enable_amp is False, 'stylegan2_d cannot supoort amp training!'
        assert not (add_scene_d and add_object_head), 'add_scene_d and add_object_head cannot be true together!'
        # if not use_stylegan2_d: assert self.config.enable_amp is True , 'pigan_d supoort amp training!'
        batch_split = 2
        if use_sr:
            nerf_feature_dim = 32 if nerf_feature_dim <= 3 else nerf_feature_dim
            batch_split = 1

        d_fmaps_base = int(self.args.pop('d_fmaps_factor') * (32 << 10))
        g_fmaps_base = int(self.args.pop('g_fmaps_factor') * (32 << 10))
        triplane_fmaps_base = int(self.args.pop('triplane_fmaps_factor') * (32 << 10))
        impl = self.args.pop('impl')
        num_fp16_res = self.args.pop('num_fp16_res')

        # Parse network settings and training settings.
        if not isinstance(num_fp16_res, int) or num_fp16_res <= 0:
            d_fp16_res = None
            g_fp16_res = None
            conv_clamp = None
        else:
            d_fp16_res = max(resolution // (2 ** (num_fp16_res - 1)),
                             2)
            g_fp16_res = max(resolution // (2 ** (num_fp16_res - 1)),
                             2)
            conv_clamp = 256

        # Parse data transformation settings.
        data_transform_config = dict(
            decode=dict(transform_type='Decode',
                        image_channels=image_channels,
                        return_square=False, center_crop=False),
            # resize1=dict(transform_type='ResizeShortside',
            #             image_size=320),
            # centercrop=dict(transform_type='CenterCrop',
            #                 crop_size=256),
            resize2=dict(transform_type='Resize',
                         image_size=resolution),
            horizontal_flip=dict(transform_type='Flip',
                                 horizontal_prob=0.0,
                                 vertical_prob=0.0),
            normalize=dict(transform_type='Normalize',
                           min_val=min_val,
                           max_val=max_val)
        )

        self.config.data.train.dataset_type = DATASET
        self.config.data.train.transform_config = data_transform_config
        self.config.data.val.dataset_type = DATASET
        self.config.data.val.transform_config = data_transform_config

        self.config.data.train.use_bbox_2d = use_bbox_2d 
        self.config.data.val.use_bbox_2d = use_bbox_2d  


        self.config.data.train.enable_flip = enable_flip 
        self.config.data.val.enable_flip = False 

        self.config.data.train.num_bbox = num_bbox
        self.config.data.val.num_bbox = num_bbox

        self.config.data.train.num_classes = label_dim
        self.config.data.val.num_classes = label_dim 

        g_init_res = self.args.pop('g_init_res')
        d_init_res = 4  # This should be fixed as 4.

        latent_dim = self.args.pop('latent_dim')
        d_mbstd_groups = self.args.pop('d_mbstd_groups')

        background_only = self.args.pop('background_only')
        static_background = self.args.pop('static_background')
        background_path = self.args.pop('background_path')
        
        ray_start, ray_end = 5, 20
        ps_cfg = dict(num_steps=num_steps,
                     bg_num_steps=bg_num_steps,
                     ray_start=ray_start,
                     ray_end=ray_end,
                     radius=1,
                     horizontal_mean=PI/2,
                     horizontal_stddev=0.3,
                     vertical_mean=PI/2,
                     vertical_stddev=0.155,
                     camera_dist='gaussian',
                     fov=48.7014,
                     perturb_mode=None,
                     cam_path=cam_path,
                     voxel_size=voxel_size,
                     transform_type=ps_type)
        hs_cfg = dict(clamp_mode='relu')
        vr_cfg = dict(clamp_mode='relu')
        self.config.add_scene_d = add_scene_d

        d_lr = self.args.pop('d_lr')
        d_beta_1 = self.args.pop('d_beta_1')
        d_beta_2 = self.args.pop('d_beta_2')
        g_lr = self.args.pop('g_lr')
        g_beta_1 = self.args.pop('g_beta_1')
        g_beta_2 = self.args.pop('g_beta_2')
        r1_interval = self.args.pop('r1_interval')
        pl_interval = self.args.pop('pl_interval')
        d_architecture = self.args.pop('d_architecture'),
        g_architecture = self.args.pop('g_architecture'),

        if r1_interval is not None and r1_interval > 0:
            d_mb_ratio = r1_interval / (r1_interval + 1)
            d_lr = d_lr * d_mb_ratio
            d_beta_1 = d_beta_1 ** d_mb_ratio
            d_beta_2 = d_beta_2 ** d_mb_ratio
        if pl_interval is not None and pl_interval > 0:
            g_mb_ratio = pl_interval / (pl_interval + 1)
            g_lr = g_lr * g_mb_ratio
            g_beta_1 = g_beta_1 ** g_mb_ratio
            g_beta_2 = g_beta_2 ** g_mb_ratio

        self.config.models.update(
            discriminator=dict(
                model=dict(model_type=DISCRIMINATOR,
                           resolution=resolution,
                           image_channels=image_channels*2 if dual_dist else image_channels,
                           init_res=4,
                           label_dim=0,
                           # architecture=self.args.pop('d_architecture'),
                           fmaps_base=d_fmaps_base,
                           conv_clamp=conv_clamp,
                           mbstd_groups=d_mbstd_groups,
                           add_coords=d_add_coords,
                           add_object_head=add_object_head,
                           use_pg=use_pg),
                lr=dict(lr_type='FIXED'),
                opt=dict(opt_type='Adam',
                         base_lr=d_lr,
                         betas=(d_beta_1, d_beta_2)),
                kwargs_train=dict(fp16_res=d_fp16_res, impl=impl),
                kwargs_val=dict(fp16_res=None, impl=impl),
                has_unused_parameters=True,
            ),
            generator=dict(
                model=dict(model_type=GENERATOR,
                           resolution=resolution,
                           # image_channels=image_channels,
                           z_dim=latent_dim,
                           w_dim=latent_dim,
                           label_dim=label_dim,
                           repeat_w=True,
                           mapping_layers=self.args.pop('g_num_mappings'),
                           synthesis_input_dim=3,
                           synthesis_output_dim=256,
                           synthesis_layers=8,
                           # grid_scale=grid_scale,
                           num_bbox=num_bbox,
                           background_only=background_only,
                           static_background=static_background,
                           background_path=background_path,
                           use_object=use_object,
                           use_hs=use_hs,
                           fg_use_dirs=self.args.pop('fg_use_dirs'),
                           bg_use_dirs=self.args.pop('bg_use_dirs'),
                           ps_cfg=ps_cfg,
                           hs_cfg=hs_cfg,
                           vr_cfg=vr_cfg,
                           nerf_res=nerf_resolution,
                           bg_nerf_res=bg_nerf_resolution,
                           use_sr=use_sr,
                           fmaps_base=g_fmaps_base,
                           conv_clamp=conv_clamp,
                           feature_dim=nerf_feature_dim,
                           condition_sample=condition_sample,
                           bg_condition_bbox=bg_condition_bbox,
                           fg_condition_bbox=fg_condition_bbox,
                           condition_bbox_embed_cfg=dict(input_dim=3,
                                                         max_freq_log2=n_freq-1,
                                                         N_freqs=n_freq),
                           use_mask=use_mask,
                           noise_type=noise_type,
                           kernel_size=kernel_size,
                           use_bbox_label=use_bbox_label,
                           use_triplane=use_triplane,
                           triplane_cfg= dict(resolution=128,
                                              image_channels=32*3,
                                              init_res=g_init_res,
                                              z_dim=latent_dim,
                                              label_dim=0,
                                              mapping_layers=1,
                                              architecture='skip',
                                              fmaps_base=triplane_fmaps_base,
                                              conv_clamp=conv_clamp),
                           ),
                lr=dict(lr_type='FIXED'),
                opt=dict(opt_type='Adam',
                         base_lr=g_lr,
                         betas=(g_beta_1, g_beta_2)),
                # Please turn off `fused_modulate` during training, which is
                # because the customized gradient computation omits weights, and
                # the fused operation will introduce division by 0.
                kwargs_train=dict(
                    w_moving_decay=self.args.pop('w_moving_decay'),
                    sync_w_avg=self.args.pop('sync_w_avg'),
                    style_mixing_prob=self.args.pop('style_mixing_prob'),
                    noise_mode='random',
                    fused_modulate=False,
                    fp16_res=g_fp16_res,
                    impl=impl),
                kwargs_val=dict(noise_mode='const',
                                fused_modulate=False,
                                fp16_res=None,
                                impl=impl),
                g_ema_img=self.args.pop('g_ema_img'),
                g_ema_rampup=self.args.pop('g_ema_rampup'),
                has_unused_parameters=True
            )
        )
        if add_scene_d:
            if not use_mask:
                img_c = image_channels
            else:
                img_c = image_channels + 1
            if dual_dist:
                img_c += image_channels
            discriminator_object = dict(
                                        model=dict(model_type='StyleGAN2Discriminator',
                                                   resolution=object_resolution,
                                                   image_channels=img_c,
                                                   init_res=4,
                                                   label_dim=label_dim,
                                                   # architecture=self.args.pop('d_architecture'),
                                                   fmaps_base=d_fmaps_base,
                                                   conv_clamp=conv_clamp,
                                                   mbstd_groups=d_mbstd_groups,
                                                   add_coords=d_add_coords,
                                                   optimize_prob=d_optimize_prob,
                                                   use_pg=object_use_pg),
                                        lr=dict(lr_type='FIXED'),
                                        opt=dict(opt_type='Adam',
                                                 base_lr=d_lr,
                                                 betas=(d_beta_1, d_beta_2)),
                                        kwargs_train=dict(fp16_res=d_fp16_res, impl=impl),
                                        kwargs_val=dict(fp16_res=None, impl=impl),
                                        has_unused_parameters=True,
                                    )
            self.config.models.update(
                discriminator_object=discriminator_object,
                )
        if LOSS == 'DiscoSceneLoss':
            pl_batch_shrink=self.args.pop('pl_batch_shrink'),
            pl_weight=self.args.pop('pl_weight'),
            pl_decay=self.args.pop('pl_decay'),
            self.config.loss.update(
                loss_type=LOSS,
                d_loss_kwargs=dict(r1_gamma=self.args.pop('r1_gamma'),
                                   latent_gamma=0,
                                   position_gamma=0,
                                   batch_split=batch_split,
                                   fade_steps=10000,
                                   scene_gamma=scene_gamma,
                                   object_gamma=object_gamma,
                                   ),
                g_loss_kwargs=dict(entropy_gamma=entropy_gamma,
                                   distortion_gamma=distortion_gamma,
                                   top_k_interval=2000,
                                   top_v=top_v),
                use_object=use_object,
                add_scene_d=add_scene_d,
                bbox_scale=bbox_scale,
                object_use_pg=object_use_pg,
                object_use_ada=object_use_ada,
                pad_object=pad_object,
                use_mask=use_mask,
                dual_dist=dual_dist,
            )
        else:
            raise NotImplementedError

        self.config.controllers.update(
            DatasetVisualizer=dict(
                viz_keys=['raw_image'],
                viz_num=(32 if label_dim == 0 else 8),
                viz_name='Real Data',
                viz_groups=(4 if label_dim == 0 else 1),
                viz_classes=min(10, label_dim),
                row_major=True,
                min_val=min_val,
                max_val=max_val,
                shuffle=False
            )
        )
        if use_pg:
            if use_small_lr:
                lr_schedule = dict(res64=1 / 2, res128=3 / 4, res256=1)
            else:
                lr_schedule = dict(res64=1, res128=1, res256=1)

            self.config.controllers.update(
                ProgressScheduler=dict(
                    init_res=nerf_resolution,
                    final_res=resolution,
                    minibatch_repeats=1,
                    # lod_training_img=4000_000,
                    # lod_training_img=2500_000 if not debug else 80,
                    # lod_transition_img=2500_000 if not debug else 80,
                    lod_training_img=pg_imgs if not debug else 80,
                    lod_transition_img=pg_imgs if not debug else 80,
                    batch_size_schedule=dict(res32=batch_size, res64=batch_size, res128=batch_size),
                    # lr_schedule=dict(res64=3e-2, res128=3e-1, res256=1),
                    reset_optimizer=reset_optimizer,
                    lr_schedule=lr_schedule,
                )
            )

        if self.args.pop('use_ada'):
            self.config.aug.update(
                aug_type='AdaAug',
                # Default augmentation strategy adopted by StyleGAN2-ADA.
                xflip=1,
                rotate90=1,
                xint=1,
                scale=1,
                rotate=1,
                aniso=1,
                xfrac=1,
                brightness=1,
                contrast=1,
                lumaflip=0,
                hue=0,
                saturation=1,
                imgfilter=0,
                noise=0,
                cutout=0
            )
            if ada_w_color:
                self.config.aug.update(lumaflip=1,
                                       hue=1,)
            self.config.aug_kwargs.update(impl='cuda')
            self.config.controllers.update(
                AdaAugController=dict(
                    every_n_iters=4,
                    init_p=0.0,
                    target_p=ada_target_p,
                    speed_img=500_000,
                    strategy=ada_type,
                    milestone=ada_milestone,
                )
            )
        if object_use_ada:
            self.config.object_aug.update(
                aug_type='AdaAug',
                # Default augmentation strategy adopted by StyleGAN2-ADA.
                xflip=1,
                rotate90=0,
                xint=0,
                scale=1,
                rotate=0,
                aniso=1,
                xfrac=0,
                brightness=1,
                contrast=1,
                lumaflip=1,
                hue=1,
                saturation=1,
                imgfilter=0,
                noise=0,
                cutout=0
            )
            if objectada_w_spatial:
                self.config.object_aug.update(rotate=1,
                                       xflip=1,
                                       xint=1,
                                       scale=1,
                                       rotate90=1,
                                       aniso=1,
                                       xfrac=1,
                                       )
            self.config.object_aug_kwargs.update(impl='cuda')
            self.config.controllers.update(
                ObjectAdaAugController=dict(
                    every_n_iters=4,
                    init_p=0.0,
                    target_p=object_ada_target_p,
                    speed_img=500_000,
                    strategy=object_ada_type,
                    milestone=object_ada_milestone,
                )
            )
             
        self.config.metrics.update(
            FID50KFull=dict(
                init_kwargs=dict(name='fid50k',
                                 latent_dim=(num_bbox + 1, latent_dim),
                                 bbox_as_input=True,
                                 label_dim=label_dim),
                eval_kwargs=dict(
                    generator_smooth=dict(fused_modulate=False,
                                          fp16_res=None,
                                          impl=impl),
                ),
                interval=None,
                first_iter=False,
                save_best=True
            ),
            GANSnapshot=dict(
                init_kwargs=dict(name='snapshot',
                                 latent_dim=(num_bbox + 1, latent_dim),
                                 latent_num=32,
                                 bbox_as_input=True,
                                 label_dim=label_dim,
                                 min_val=min_val,
                                 max_val=max_val),
                eval_kwargs=dict(
                    generator_smooth=dict(fused_modulate=False,
                                          fp16_res=None,
                                          impl=impl),
                ),
                interval=1000,
                first_iter=None,
                save_best=False
            )
        )
