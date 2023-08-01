# python3.7
"""Configuration for training StyleGAN3."""

from .base_config import BaseConfig

__all__ = ['StyleGAN3Config']

RUNNER = 'StyleGAN3Runner'
DATASET = 'ImageDataset'
DISCRIMINATOR = 'StyleGAN2Discriminator'
GENERATOR = 'StyleGAN3Generator'
LOSS = 'StyleGAN3Loss'


class StyleGAN3Config(BaseConfig):
    """Defines the configuration for training StyleGAN3."""

    name = 'stylegan3'
    hint = 'Train a StyleGAN3 model.'
    info = '''
To train a StyleGAN3 model, the recommended settings are as follows:

\b
- batch_size: 4 (for FF-HQ-U dataset, 8 GPU)
- val_batch_size: 16 (for FF-HQ-U dataset, 8 GPU)
- data_repeat: 200 (for FF-HQ-U dataset)
- total_img: 25_000_000 (for FF-HQ-U dataset)
- train_data_mirror: True (for FF-HQ-U dataset)
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
                '--g_kernel_size', type=cls.int_type, default=3,
                help='Kernel size of the convolutional layers in generator.'),
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
                '--g_num_mappings', type=cls.int_type, default=2,
                help='Number of mapping layers of generator.'),
            cls.command_option(
                '--impl', type=str, default='cuda',
                help='Control the implementation of some neural operations.'),
            cls.command_option(
                '--num_fp16_res', type=cls.int_type, default=0,
                help='Number of (highest) resolutions that use `float16` '
                     'precision for training, which speeds up the training yet '
                     'barely affects the performance. The official StyleGAN3 '
                     'uses 4 by default.')
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
                '--w_moving_decay', type=cls.float_type, default=0.998,
                help='Decay factor for updating `w_avg`.'),
            cls.command_option(
                '--sync_w_avg', type=cls.bool_type, default=False,
                help='Synchronizing the update of `w_avg` across replicas.'),
            cls.command_option(
                '--style_mixing_prob', type=cls.float_type, default=0.0,
                help='Probability to perform style mixing as a training '
                     'regularization.'),
            cls.command_option(
                '--r1_gamma', type=cls.float_type, default=10.0,
                help='Factor to control the strength of gradient penalty.'),
            cls.command_option(
                '--r1_interval', type=cls.int_type, default=16,
                help='Interval (in iterations) to perform gradient penalty.'),
            cls.command_option(
                '--blur_init_sigma', type=cls.float_type, default=0.0,
                help='Factor to control the image blurry shown to the '
                     'discriminator.'),
            cls.command_option(
                '--blur_fade_img', type=cls.int_type, default=0,
                help='Factor to control the fade-out of image blurry.'),
            cls.command_option(
                '--pl_batch_shrink', type=cls.int_type, default=2,
                help='Factor to reduce the batch size for perceptual path '
                     'length regularization.'),
            cls.command_option(
                '--pl_weight', type=cls.float_type, default=0.0,
                help='Factor to control the strength of perceptual path length '
                     'regularization.'),
            cls.command_option(
                '--pl_decay', type=cls.float_type, default=0.01,
                help='Decay factor for perceptual path length regularization.'),
            cls.command_option(
                '--pl_interval', type=cls.int_type, default=0,
                help='Interval (in iterations) to perform perceptual path '
                     'length regularization.'),
            cls.command_option(
                '--g_ema_img', type=cls.int_type, default=10_000,
                help='Factor for updating the smoothed generator, which is '
                     'particularly used for inference.'),
            cls.command_option(
                '--g_ema_rampup', type=cls.float_type, default=0.05,
                help='Rampup factor for updating the smoothed generator, which '
                     'is particularly used for inference. Set as `0` to '
                     'disable warming up.'),
            cls.command_option(
                '--use_ada', type=cls.bool_type, default=False,
                help='Whether to use adaptive augmentation pipeline.')
        ])

        return options

    @classmethod
    def get_recommended_options(cls):
        recommended_opts = super().get_recommended_options()
        recommended_opts.extend([
            'resolution', 'num_fp16_res', 'latent_dim', 'label_dim', 'g_lr',
            'd_lr', 'g_kernel_size', 'd_fmaps_factor', 'd_mbstd_groups',
            'g_fmaps_factor', 'g_num_mappings', 'r1_interval', 'pl_interval',
            'blur_init_sigma', 'blur_fade_img', 'r1_gamma', 'pl_weight',
            'use_ada', 'style_mixing_prob', 'g_ema_img',
        ])
        return recommended_opts

    def parse_options(self):
        super().parse_options()

        resolution = self.args.pop('resolution')
        image_channels = self.args.pop('image_channels')
        min_val = self.args.pop('min_val')
        max_val = self.args.pop('max_val')

        # Parse data transformation settings.
        data_transform_kwargs = dict(
            image_size=resolution,
            image_channels=image_channels,
            min_val=min_val,
            max_val=max_val
        )
        self.config.data.train.dataset_type = DATASET
        self.config.data.train.transform_kwargs = data_transform_kwargs
        self.config.data.val.dataset_type = DATASET
        self.config.data.val.transform_kwargs = data_transform_kwargs

        g_kernel_size = self.args.pop('g_kernel_size')
        d_init_res = 4  # This should be fixed as 4.
        latent_dim = self.args.pop('latent_dim')
        label_dim = self.args.pop('label_dim')
        d_fmaps_base = int(self.args.pop('d_fmaps_factor') * (32 << 10))
        g_fmaps_base = int(self.args.pop('g_fmaps_factor') * (32 << 10))
        g_fmaps_max = 512
        g_use_radial_filter = False
        impl = self.args.pop('impl')
        num_fp16_res = self.args.pop('num_fp16_res')

        # Adjust settings for config-R.
        if g_kernel_size == 1:
            g_fmaps_base = g_fmaps_base * 2
            g_fmaps_max = g_fmaps_max * 2
            g_use_radial_filter = True

        # Parse network settings and training settings.
        if not isinstance(num_fp16_res, int) or num_fp16_res <= 0:
            d_fp16_res = None
            g_fp16_res = None
            conv_clamp = None
        else:
            d_fp16_res = max(resolution // (2 ** (num_fp16_res - 1)),
                             d_init_res * 2)
            g_fp16_res = resolution // (2 ** (num_fp16_res - 1))
            conv_clamp = 256

        d_lr = self.args.pop('d_lr')
        d_beta_1 = self.args.pop('d_beta_1')
        d_beta_2 = self.args.pop('d_beta_2')
        g_lr = self.args.pop('g_lr')
        g_beta_1 = self.args.pop('g_beta_1')
        g_beta_2 = self.args.pop('g_beta_2')
        r1_interval = self.args.pop('r1_interval')
        pl_interval = self.args.pop('pl_interval')

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
                           image_channels=image_channels,
                           init_res=d_init_res,
                           label_dim=label_dim,
                           fmaps_base=d_fmaps_base,
                           conv_clamp=conv_clamp,
                           mbstd_groups=self.args.pop('d_mbstd_groups')),
                lr=dict(lr_type='FIXED'),
                opt=dict(opt_type='Adam',
                         base_lr=d_lr,
                         betas=(d_beta_1, d_beta_2)),
                kwargs_train=dict(fp16_res=d_fp16_res, impl=impl),
                kwargs_val=dict(fp16_res=None, impl=impl)
            ),
            generator=dict(
                model=dict(model_type=GENERATOR,
                           resolution=resolution,
                           image_channels=image_channels,
                           kernel_size=g_kernel_size,
                           z_dim=latent_dim,
                           label_dim=label_dim,
                           mapping_layers=self.args.pop('g_num_mappings'),
                           fmaps_base=g_fmaps_base,
                           fmaps_max=g_fmaps_max,
                           use_radial_filter=g_use_radial_filter,
                           conv_clamp=conv_clamp),
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
                    fp16_res=g_fp16_res,
                    impl=impl),
                kwargs_val=dict(fp16_res=None, impl=impl),
                g_ema_img=self.args.pop('g_ema_img'),
                g_ema_rampup=self.args.pop('g_ema_rampup')
            )
        )

        self.config.loss.update(
            loss_type=LOSS,
            d_loss_kwargs=dict(r1_gamma=self.args.pop('r1_gamma'),
                               r1_interval=r1_interval,
                               blur_init_sigma=self.args.pop('blur_init_sigma'),
                               blur_fade_img=self.args.pop('blur_fade_img')),
            g_loss_kwargs=dict(pl_batch_shrink=self.args.pop('pl_batch_shrink'),
                               pl_weight=self.args.pop('pl_weight'),
                               pl_decay=self.args.pop('pl_decay'),
                               pl_interval=pl_interval)
        )

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
                lumaflip=1,
                hue=1,
                saturation=1,
                imgfilter=0,
                noise=0,
                cutout=0
            )
            self.config.aug_kwargs.update(impl='cuda')
            self.config.controllers.update(
                AdaAugController=dict(
                    every_n_iters=4,
                    init_p=0.0,
                    target_p=0.6,
                    speed_img=500_000,
                    strategy='adaptive'
                )
            )

        self.config.metrics.update(
            FID50KFull=dict(
                init_kwargs=dict(name='fid50k_full',
                                 latent_dim=latent_dim,
                                 label_dim=label_dim),
                eval_kwargs=dict(
                    generator_smooth=dict(fp16_res=None, impl=impl),
                ),
                interval=None,
                first_iter=None,
                save_best=True
            ),
            GANSnapshot=dict(
                init_kwargs=dict(name='snapshot',
                                 latent_dim=latent_dim,
                                 latent_num=32,
                                 label_dim=label_dim,
                                 min_val=min_val,
                                 max_val=max_val),
                eval_kwargs=dict(
                    generator_smooth=dict(fp16_res=None, impl=impl),
                ),
                interval=None,
                first_iter=None,
                save_best=False
            )
        )
