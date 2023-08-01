# python3.7
"""Test metrics.

NOTE: This file can be used as an example for distributed inference/evaluation.
This file only supports testing GAN related metrics (including FID, IS, KID,
GAN precision-recall, saving snapshot, and equivariance) by loading a
pre-trained generator. To test more metrics, please customize your own script.
"""

import argparse

import torch

from datasets import build_dataset
from models import build_model
from metrics import build_metric
from utils.loggers import build_logger
from utils.parsing_utils import parse_bool
from utils.parsing_utils import parse_json
from utils.dist_utils import init_dist
from utils.dist_utils import exit_dist


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='Run metric test.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to the dataset used for metric computation.')
    parser.add_argument('--dataset_ann', type=str, required=True,
                        help='Path to the dataset used for metric computation.')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the pre-trained model weights.')
    parser.add_argument('--G_kwargs', type=parse_json, default={},
                        help='Runtime keyword arguments for generator. Please '
                             'wrap the argument into single quotes with '
                             'keywords in double quotes. Beside, remove any '
                             'whitespace to avoid mis-parsing. For example, to '
                             'turn on truncation with probability 0.5 on 2 '
                             'layers, pass `--G_kwargs \'{"trunc_psi":0.5,'
                             '"trunc_layers":2}\'`. (default: %(default)s)')
    parser.add_argument('--work_dir', type=str,
                        default='work_dirs/metric_tests',
                        help='Working directory for metric test. (default: '
                             '%(default)s)')
    parser.add_argument('--real_num', type=int, default=-1,
                        help='Number of real data used for testing. Negative '
                             'means using all data. (default: %(default)s)')
    parser.add_argument('--fake_num', type=int, default=1000,
                        help='Number of fake data used for testing. (default: '
                             '%(default)s)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size used for metric computation. '
                             '(default: %(default)s)')
    parser.add_argument('--test_fid', type=parse_bool, default=False,
                        help='Whether to test FID. (default: %(default)s)')
    parser.add_argument('--test_is', type=parse_bool, default=False,
                        help='Whether to test IS. (default: %(default)s)')
    parser.add_argument('--test_kid', type=parse_bool, default=False,
                        help='Whether to test KID. (default: %(default)s)')
    parser.add_argument('--test_gan_pr', type=parse_bool, default=False,
                        help='Whether to test GAN precision-recall. '
                             '(default: %(default)s)')
    parser.add_argument('--test_snapshot', type=parse_bool, default=False,
                        help='Whether to test saving snapshot. '
                             '(default: %(default)s)')
    parser.add_argument('--test_equivariance', type=parse_bool, default=False,
                        help='Whether to test GAN Equivariance. '
                             '(default: %(default)s)')
    parser.add_argument('--launcher', type=str, default='pytorch',
                        choices=['pytorch', 'slurm'],
                        help='Distributed launcher. (default: %(default)s)')
    parser.add_argument('--backend', type=str, default='nccl',
                        choices=['nccl', 'gloo', 'mpi'],
                        help='Distributed backend. (default: %(default)s)')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Replica rank on the current node. This field is '
                             'required by `torch.distributed.launch`. '
                             '(default: %(default)s)')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Initialize distributed environment.
    init_dist(launcher=args.launcher, backend=args.backend)

    # CUDNN settings.
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    state = torch.load(args.model, map_location='cpu')
    G = build_model(**state['model_kwargs_init']['generator_smooth'])
    G.load_state_dict(state['models']['generator_smooth'])
    G.eval().cuda()

    data_transform_kwargs = dict(
        image_size=G.resolution, image_channels=G.image_channels)
    dataset_kwargs = dict(dataset_type='ImageDataset',
                          root_dir=args.dataset,
                          annotation_path=args.dataset_ann,
                          annotation_meta=None,
                          max_samples=args.real_num,
                          mirror=False,
                          file_format=None,
                          annotation_format=None,
                          transform_kwargs=data_transform_kwargs)
    data_loader_kwargs = dict(data_loader_type='iter',
                              repeat=1,
                              num_workers=4,
                              prefetch_factor=2,
                              pin_memory=True)
    data_loader = build_dataset(for_training=False,
                                batch_size=args.batch_size,
                                dataset_kwargs=dataset_kwargs,
                                data_loader_kwargs=data_loader_kwargs)

    if torch.distributed.get_rank() == 0:
        logger = build_logger('normal', logfile=None, verbose_log=True)
    else:
        logger = build_logger('dummy')

    real_num = (len(data_loader.dataset)
                if args.real_num < 0 else args.real_num)
    if args.test_fid:
        logger.info('========== Test FID ==========')
        metric = build_metric('FID',
                              name=f'fid{args.fake_num}_real{real_num}',
                              work_dir=args.work_dir,
                              logger=logger,
                              batch_size=args.batch_size,
                              latent_dim=G.latent_dim,
                              label_dim=G.label_dim,
                              real_num=args.real_num,
                              fake_num=args.fake_num)
        result = metric.evaluate(data_loader, G, args.G_kwargs)
        metric.save(result)
    if args.test_is:
        logger.info('========== Test IS ==========')
        metric = build_metric('IS',
                              name=f'is{args.fake_num}',
                              work_dir=args.work_dir,
                              logger=logger,
                              batch_size=args.batch_size,
                              latent_dim=G.latent_dim,
                              label_dim=G.label_dim,
                              latent_num=args.fake_num)
        result = metric.evaluate(data_loader, G, args.G_kwargs)
        metric.save(result)
    if args.test_kid:
        logger.info('========== Test KID ==========')
        metric = build_metric('KID',
                              name=f'kid{args.fake_num}_real{real_num}',
                              work_dir=args.work_dir,
                              logger=logger,
                              batch_size=args.batch_size,
                              latent_dim=G.latent_dim,
                              label_dim=G.label_dim,
                              real_num=args.real_num,
                              fake_num=args.fake_num)
        result = metric.evaluate(data_loader, G, args.G_kwargs)
        metric.save(result)
    if args.test_gan_pr:
        logger.info('========== Test GAN PR ==========')
        metric = build_metric('GANPR',
                              name=f'pr{args.fake_num}_real{real_num}',
                              work_dir=args.work_dir,
                              logger=logger,
                              batch_size=args.batch_size,
                              latent_dim=G.latent_dim,
                              label_dim=G.label_dim,
                              real_num=args.real_num,
                              fake_num=args.fake_num)
        result = metric.evaluate(data_loader, G, args.G_kwargs)
        metric.save(result)
    if args.test_snapshot:
        logger.info('========== Test GAN Snapshot ==========')
        metric = build_metric('GANSnapshot',
                              name='snapshot',
                              work_dir=args.work_dir,
                              logger=logger,
                              batch_size=args.batch_size,
                              latent_dim=G.latent_dim,
                              label_dim=G.label_dim,
                              latent_num=min(args.fake_num, 50))
        result = metric.evaluate(data_loader, G, args.G_kwargs)
        metric.save(result)
    if args.test_equivariance:
        logger.info('========== Test GAN Equivariance ==========')
        metric = build_metric('Equivariance',
                              name='equivariance',
                              work_dir=args.work_dir,
                              logger=logger,
                              batch_size=args.batch_size,
                              latent_dim=G.latent_dim,
                              label_dim=G.label_dim,
                              latent_num=args.fake_num,
                              test_eqt=True,
                              test_eqt_frac=True,
                              test_eqr=True)
        result = metric.evaluate(data_loader, G, args.G_kwargs)
        metric.save(result)

    # Exit distributed environment.
    exit_dist()


if __name__ == '__main__':
    main()
