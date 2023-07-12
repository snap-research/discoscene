# python3.7
"""Configuration for training StyleGAN."""

from .base_config import BaseConfig

__all__ = ['StyleGANConfig']

RUNNER = 'StyleGANRunner'
DATASET = 'ImageDataset'
DISCRIMINATOR = 'StyleGANDiscriminator'
GENERATOR = 'StyleGANGenerator'
LOSS = 'StyleGANLoss'


class StyleGANConfig(BaseConfig):
    """Defines the configuration for training StyleGAN."""

    name = 'stylegan'
    hint = 'Train a StyleGAN model.'
    info = '''
To train a StyleGAN model, the recommended settings are as follows:

\b
- batch_size: 4 (for FF-HQ dataset, 8 GPU)
- val_batch_size: 16 (for FF-HQ dataset, 8 GPU)
- data_repeat: 200 (for FF-HQ dataset)
- total_img: 25_000_000 (for FF-HQ dataset)
- train_data_mirror: True (for FF-HQ dataset)
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
                     'discriminator, which will be `factor * 16384`.'),
            cls.command_option(
                '--d_mbstd_groups', type=cls.int_type, default=4,
                help='Number of groups for MiniBatchSTD layer of '
                     'discriminator.'),
            cls.command_option(
                '--g_fmaps_factor', type=cls.float_type, default=1.0,
                help='A factor to control the number of feature maps of '
                     'generator, which will be `factor * 16384`.'),
            cls.command_option(
                '--g_num_mappings', type=cls.int_type, default=8,
                help='Number of mapping layers of generator.')
        ])

        options['Training settings'].extend([
            cls.command_option(
                '--d_lr', type=cls.float_type, default=0.001,
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
                '--g_lr', type=cls.float_type, default=0.001,
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
                '--g_ema_img', type=cls.int_type, default=10_000,
                help='Factor for updating the smoothed generator, which is '
                     'particularly used for inference.'),
            cls.command_option(
                '--use_ada', type=cls.bool_type, default=False,
                help='Whether to use adaptive augmentation pipeline.')
        ])

        return options

    @classmethod
    def get_recommended_options(cls):
        recommended_opts = super().get_recommended_options()
        recommended_opts.extend([
            'resolution', 'latent_dim', 'label_dim', 'g_lr', 'd_lr',
            'd_fmaps_factor', 'd_mbstd_groups', 'g_fmaps_factor',
            'g_num_mappings', 'g_ema_img', 'style_mixing_prob',
            'r1_gamma', 'use_ada'
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

        g_init_res = self.args.pop('g_init_res')
        d_init_res = 4  # This should be fixed as 4.
        latent_dim = self.args.pop('latent_dim')
        label_dim = self.args.pop('label_dim')
        d_fmaps_base = int(self.args.pop('d_fmaps_factor') * (16 << 10))
        g_fmaps_base = int(self.args.pop('g_fmaps_factor') * (16 << 10))

        self.config.models.update(
            discriminator=dict(
                model=dict(model_type=DISCRIMINATOR,
                           resolution=resolution,
                           image_channels=image_channels,
                           init_res=d_init_res,
                           label_dim=label_dim,
                           fmaps_base=d_fmaps_base,
                           mbstd_groups=self.args.pop('d_mbstd_groups')),
                lr=dict(lr_type='FIXED'),
                opt=dict(opt_type='Adam',
                         base_lr=self.args.pop('d_lr'),
                         betas=(self.args.pop('d_beta_1'),
                                self.args.pop('d_beta_2'))),
                kwargs_train=dict(enable_amp=self.config.enable_amp),
                kwargs_val=dict(enable_amp=False),
                has_unused_parameters=True
            ),
            generator=dict(
                model=dict(model_type=GENERATOR,
                           resolution=resolution,
                           image_channels=image_channels,
                           init_res=g_init_res,
                           z_dim=latent_dim,
                           label_dim=label_dim,
                           mapping_layers=self.args.pop('g_num_mappings'),
                           fmaps_base=g_fmaps_base),
                lr=dict(lr_type='FIXED'),
                opt=dict(opt_type='Adam',
                         base_lr=self.args.pop('g_lr'),
                         betas=(self.args.pop('g_beta_1'),
                                self.args.pop('g_beta_2'))),
                kwargs_train=dict(
                    w_moving_decay=self.args.pop('w_moving_decay'),
                    sync_w_avg=self.args.pop('sync_w_avg'),
                    style_mixing_prob=self.args.pop('style_mixing_prob'),
                    noise_mode='random',
                    enable_amp=self.config.enable_amp),
                kwargs_val=dict(noise_mode='const', enable_amp=False),
                g_ema_img=self.args.pop('g_ema_img'),
                has_unused_parameters=True
            )
        )

        self.config.loss.update(
            loss_type=LOSS,
            d_loss_kwargs=dict(r1_gamma=self.args.pop('r1_gamma')),
            g_loss_kwargs=dict()
        )

        self.config.controllers.update(
            ProgressScheduler=dict(
                init_res=g_init_res * 2,
                final_res=resolution,
                minibatch_repeats=4,
                lod_training_img=600_000,
                lod_transition_img=600_000,
                batch_size_schedule=dict(res4=64, res8=32, res16=16, res32=8)
            ),
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
                    generator_smooth=dict(noise_mode='random',
                                          enable_amp=False),
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
                    generator_smooth=dict(noise_mode='const',
                                          enable_amp=False),
                ),
                interval=None,
                first_iter=None,
                save_best=False
            )
        )
