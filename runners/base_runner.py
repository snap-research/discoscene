# python3.7
"""Contains the base class for runner.

The runner is particularly used for distributed training.
"""

import os
import json
from copy import deepcopy
import random
import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.collect_env import get_pretty_env_info

from datasets import build_dataset
from models import build_model
from metrics import build_metric
from utils.file_transmitters import build_file_transmitter
from utils.loggers import build_logger
from utils.dist_utils import ddp_sync
from utils.formatting_utils import format_time
from utils.tf_utils import import_tb_writer
from .augmentations import build_aug
from .controllers import build_controller
from .losses import build_loss
from .utils.optimizer import build_optimizer
from .utils.running_stats import RunningStats
from .utils.profiler import Profiler
from .utils.freezer import Freezer

SummaryWriter = import_tb_writer()

__all__ = ['BaseRunner']


class BaseRunner(object):
    """Defines the base runner class."""

    def __init__(self, config):
        assert dist.is_initialized(), 'Distributed environment is required!'

        self._name = self.__class__.__name__
        self._config = deepcopy(config)
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        self._device = torch.cuda.current_device()
        self._config.num_gpus = self.world_size
        self.is_chief = (self.rank == 0)
        self.interactive = self.config.interactive

        assert self.config.job_name, 'Job name is required!'
        self._job_name = self.config.job_name
        assert self.config.work_dir, 'Working directory is required!'
        self._work_dir = self.config.work_dir
        self.build_work_dir()

        # Set up working directory, logger, TensorBoard, file transmitter, etc.
        self.logger = None  # Logger.
        self.tb_writer = None  # TensorBoard writer.
        self.ft = None  # File transmitter.
        self.build_logger()
        self.build_tensorboard()
        self.build_file_transmitter()

        self.log_basic_info()
        self.set_cudnn()
        self.set_seed()
        self.prefetch_data()

        # Set up basic configurations, like batch size, training iters, etc.
        self.batch_size = self.config.batch_size
        self.val_batch_size = self.config.val_batch_size
        self._iter = 0
        self._start_iter = 0
        self.seen_img = 0
        self.total_iters = self.config.total_iters
        self.total_epochs = self.config.total_epochs
        self.total_img = self.config.total_img
        if self.total_iters <= 0 and self.total_img > 0:  # pylint: disable=chained-comparison
            self.total_iters = int(self.total_img / self.minibatch + 0.5)

        # Initialize empty dataset loader and augmentation pipeline.
        self.train_loader = None
        self.val_loader = None
        self.augment = None
        self.batch_data = dict()  # Record the current batch for viz ONLY.

        # Initialize empty models, opts, lrs, loss.
        self.models = dict()
        self.ddp_models = dict()
        self.model_kwargs_init = dict()  # Arguments to initialize the model.
        self.model_kwargs_train = dict()  # Arguments for training `forward()`.
        self.model_kwargs_val = dict()  # Arguments for validation `forward()`.
        self.model_has_unused_param = dict()  # Whether has unused parameter.
        self.model_broadcast_buffers = dict()  # Whether to broadcast buffers.
        self.opt_config = dict()
        self.optimizers = dict()
        self.lr_config = dict()
        self.lr_schedulers = dict()
        self.loss = None

        # Initialize empty controllers.
        self.controllers = []

        # Initialize empty evaluation metrics.
        self.metrics = dict()
        self.eval_results = dict()

        # Initialize empty statistic information.
        self.running_stats = RunningStats()
        self.start_time = 0
        self.end_time = 0

        # Set up automatic mixed-precision (AMP) if needed.
        self.enable_amp = self.config.enable_amp
        self.amp_scaler = torch.cuda.amp.GradScaler(enabled=self.enable_amp)
        if self.amp_scaler.is_enabled():
            self.logger.info('Enable automatic mixed-precision training.\n')
            self.running_stats.add('Misc/AMP Scale',
                                   log_format='6.2f',
                                   log_name='amp_scale',
                                   log_strategy='CURRENT',
                                   requires_sync=False)
        else:
            self.logger.info('Disable automatic mixed-precision training.\n')

        # Build data loaders, augmentation, and convert epoch to iteration.
        self.build_train_loader()
        if len(self.config.metrics) != 0:
            self.build_val_loader()
        self.build_augment()
        if self.total_iters <= 0 and self.total_epochs > 0:  # pylint: disable=chained-comparison
            self.total_iters = self.convert_epoch_to_iter(self.total_epochs)
        if self.total_iters <= 0:
            raise ValueError(f'Total training iterations are zero!\n'
                             f'`total_iter`, `total_img`, or `total_epochs` '
                             f'should be set as positive, but '
                             f'({self.total_iters}, {self.total_img}, '
                             f'{self.total_epochs}) are received.')

        # Build models (distributed), opts, lrs, loss.
        self.build_models()
        self.log_model_info()
        self.distribute()
        self.build_optimizers()
        self.build_lr_scheduler()
        self.build_loss()

        # Build controllers.
        # NOTE: Controllers should be built after models (due to LRScheduler)
        # and augmentations (due to AdaAugController), and be built before
        # metrics (because the TensorBoard writer may be redirected by
        # controllers, yet the metrics rely on TensorBoard to write results).
        self.build_controllers()

        # Build metrics.
        self.build_metrics()

        # Load checkpoint if resume/fine-tune.
        if self.config.resume_path and self.config.weight_path:
            raise ValueError('Resume checkpoint `resume_path` and '
                             'fine-tune checkpoint `weight_path` '
                             'can not be both specified.')
        if self.config.resume_path:
            self.load(filepath=self.config.resume_path,
                      running_metadata=True,
                      optimizer=True,
                      learning_rate=True,
                      loss=True,
                      augment=True,
                      running_stats=True)
        if self.config.weight_path:
            self.load(filepath=self.config.weight_path,
                      running_metadata=False,
                      optimizer=False,
                      learning_rate=False,
                      loss=False,
                      augment=False,
                      running_stats=False)

    def close(self):
        """Closes the runner by clearing/deleting all maintained variables."""
        if self.logger is not None:
            self.logger.close()
        if self.tb_writer is not None:
            self.tb_writer.close()
        self._config = None
        self.models.clear()
        self.ddp_models.clear()
        self.model_kwargs_init.clear()
        self.model_kwargs_train.clear()
        self.model_kwargs_val.clear()
        self.model_has_unused_param.clear()
        self.model_broadcast_buffers.clear()
        self.opt_config.clear()
        self.optimizers.clear()
        self.lr_config.clear()
        self.lr_schedulers.clear()
        self.loss = None
        self.train_loader = None
        self.val_loader = None
        self.batch_data.clear()
        self.augment = None
        self.metrics.clear()
        self.eval_results.clear()
        self.running_stats = None
        self.controllers.clear()
        self.amp_scaler = None

    @property
    def name(self):
        """Returns the name of the runner."""
        return self._name

    @property
    def job_name(self):
        """Returns the job name."""
        return self._job_name

    @property
    def work_dir(self):
        """Returns the working directory of the runner."""
        return self._work_dir

    @property
    def data_dir(self):
        """Returns the data directory."""
        return os.path.join(self.work_dir, self.config.data_dir)

    @property
    def checkpoint_dir(self):
        """Returns the checkpoint directory."""
        return os.path.join(self.work_dir, self.config.checkpoint_dir)

    @property
    def result_dir(self):
        """Returns the result directory."""
        return os.path.join(self.work_dir, self.config.result_dir)

    @property
    def tensorboard_dir(self):
        """Returns the TensorBoard event directory."""
        return os.path.join(self.work_dir, self.config.tensorboard_dir)

    @property
    def profile_dir(self):
        """Returns the performance profile directory."""
        return os.path.join(self.work_dir, self.config.profile_dir)

    @property
    def resource_dir(self):
        """Returns the resource log directory."""
        return os.path.join(self.work_dir, self.config.resource_dir)

    @property
    def config_path(self):
        """Returns the path to the full configuration file."""
        return os.path.join(self.work_dir, self.config.config_filename)

    @property
    def log_data_path(self):
        """Returns the path to the log data file."""
        return os.path.join(self.work_dir, self.config.log_data_filename)

    @property
    def log_path(self):
        """Returns the path to the full log file."""
        return os.path.join(self.work_dir, self.config.log_filename)

    @property
    def config(self):
        """Returns the configuration of the runner."""
        return self._config

    @property
    def rank(self):
        """Returns the rank of the current process."""
        return self._rank

    @property
    def world_size(self):
        """Returns the world size."""
        return self._world_size

    @property
    def device(self):
        """Returns the current running device."""
        return self._device

    @property
    def iter(self):
        """Returns the current iteration."""
        return self._iter

    @property
    def start_iter(self):
        """Returns the start iteration."""
        return self._start_iter

    @property
    def minibatch(self):
        """Returns the minibatch distributed on all replicas."""
        return self.batch_size * self.world_size

    def convert_epoch_to_iter(self, epoch):
        """Converts number of epochs to number of iterations."""
        iters_per_epoch = len(self.train_loader.dataset) / self.minibatch
        return int(epoch * iters_per_epoch + 0.5)

    def build_work_dir(self):
        """Builds the working directory."""
        if self.is_chief:
            if self.interactive and os.path.exists(self.config.work_dir):
                print(f'Working directory `{self.work_dir}` has already existed!')
                while True:
                    decision = input(f'Would you like to overwrite it (Y/N): ')
                    decision = decision.strip().lower()
                    if decision == 'n':
                        raise SystemExit(f'Please specify another one.')
                    if decision == 'y':
                        print(f'Overwriting working directory `{self.work_dir}`!')
                        import shutil
                        # shutil.rmtree(f'{self.work_dir}')
                        break
                os.makedirs(self.work_dir, exist_ok=True)
                os.makedirs(self.data_dir, exist_ok=True)
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                os.makedirs(self.result_dir, exist_ok=True)
                os.makedirs(self.tensorboard_dir, exist_ok=True)
                os.makedirs(self.resource_dir, exist_ok=True)
                if self.config.enable_profiler:
                    os.makedirs(self.profile_dir, exist_ok=True)
            else:
                if os.path.exists(self.config.work_dir):
                    raise SystemExit(f'Working directory `{self.work_dir}` has '
                                     f'already existed!\n'
                                     f'Please use another job name, or otherwise '
                                     f'the results and logs may be mixed up.')
                os.makedirs(self.work_dir, exist_ok=False)
                os.makedirs(self.data_dir, exist_ok=False)
                os.makedirs(self.checkpoint_dir, exist_ok=False)
                os.makedirs(self.result_dir, exist_ok=False)
                os.makedirs(self.tensorboard_dir, exist_ok=False)
                os.makedirs(self.resource_dir, exist_ok=False)
                if self.config.enable_profiler:
                    os.makedirs(self.profile_dir, exist_ok=False)
        dist.barrier()  # Make sure the directory is built for other replicas.

    def build_logger(self):
        """Builds logger for logging messages."""
        if self.is_chief:
            logger_type = self.config.logger_type
            self.logger = build_logger(logger_type, logfile=self.log_path)
        else:
            self.logger = build_logger('dummy', logfile=None)
        dist.barrier()  # Make sure loggers for all replicas are built.

    def build_tensorboard(self):
        """Builds TensorBoard for visualizing results."""
        if SummaryWriter is None:
            self._config.use_tensorboard = False
        if self.is_chief:
            if self.config.use_tensorboard:
                self.tb_writer = SummaryWriter(self.tensorboard_dir)
            else:
                os.removedirs(self.tensorboard_dir)
        dist.barrier()  # Make sure the writer is built for other replicas.

    def build_file_transmitter(self):
        """Builds file transmitter for exchanging data with storage system."""
        if self.is_chief:
            ft_type = self.config.file_transmitter_type
            ft_kwargs = self.config.file_transmitter_kwargs
            self.ft = build_file_transmitter(ft_type, **ft_kwargs)
        else:
            self.ft = build_file_transmitter('dummy')
        dist.barrier()  # Make sure transmitters for all replicas are built.

    def log_basic_info(self):
        """Logs basic information of the current job."""
        # Log commit ID for reproducibility.
        commit_id = os.popen('git rev-parse HEAD').readline().strip()
        self.logger.info(f'Commit ID: {commit_id}\n')

        self.logger.info(f'Job name: {self.job_name}')
        self.logger.info(f'Runner type: {self.name}')
        self.logger.info(f'Working directory: {self.work_dir}\n')

        self.logger.info(f'Logger type: {self.config.logger_type}')
        self.logger.info(f'Use TensorBoard: {self.config.use_tensorboard}')
        self.logger.info(f'File system: {self.config.file_transmitter_type}\n')

        # Log running environment for reproducibility.
        self.logger.info('Running environment:\n')
        env_info = '=' * 50 + '\n'
        env_info += get_pretty_env_info()
        env_info += ('\n' + '=' * 50 + '\n')
        self.logger.print(get_pretty_env_info() + '\n')

        # Log configurations for reproducibility.
        self.logger.info('Running configuration:\n')
        config_info = '=' * 50 + '\n'
        config_info += json.dumps(self.config, indent=4).replace('"', '\'')
        config_info += ('\n' + '=' * 50 + '\n')
        self.logger.print(config_info)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)

    def set_cudnn(self):
        """Sets CUDNN backend."""
        torch.backends.cudnn.enabled = self.config.enable_cudnn
        torch.backends.cudnn.benchmark = self.config.cudnn_benchmark
        torch.backends.cudnn.deterministic = self.config.cudnn_deterministic
        torch.backends.cudnn.allow_tf32 = self.config.cudnn_allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = self.config.cudnn_allow_tf32
        self.logger.info('CUDNN settings:')
        self.logger.info(f'Enable: {torch.backends.cudnn.enabled}',
                         indent_level=1)
        self.logger.info(f'Benchmark: {torch.backends.cudnn.benchmark}',
                         indent_level=1)
        self.logger.info(f'Deterministic: {torch.backends.cudnn.deterministic}',
                         indent_level=1)
        self.logger.info(f'TensorFloat32: {torch.backends.cudnn.allow_tf32}',
                         indent_level=1)
        if self.config.cudnn_deterministic:
            self.logger.warning('CUDNN deterministic is turned on, which may '
                                'slow down the training considerably!')
        self.logger.print()

    def set_seed(self):
        """Sets random seed for reproducibility."""
        if self.config.seed >= 0:
            random.seed(self.config.seed * self.world_size + self.rank)
            np.random.seed(self.config.seed * self.world_size + self.rank)
            torch.manual_seed(self.config.seed * self.world_size + self.rank)
            self.logger.info(f'Set random seed as `{self.config.seed}`.\n')
        else:
            self.logger.warning('Do not set random seed.\n')

    def prefetch_data(self):
        """Prefetches data/checkpoint/etc. from other file system."""
        self.logger.info('Prefetching data ...')
        if self.is_chief:
            for file in self.config.prefetch_list:
                self.logger.info(f'Pulling `{file}` to `{self.data_dir}`.',
                                 indent_level=1)
                self.ft.pull(file, self.data_dir)
        dist.barrier()  # Make sure the data is prefetched for other replicas.
        self.logger.info('Finish prefetching data.\n')

    def build_models(self):
        """Builds models, optimizers, and learning rate schedulers."""
        self.logger.info('Building models ...')
        for name, config in self.config.models.items():
            self.logger.info(f'Building model `{name}` ...', indent_level=1)
            self.models[name] = build_model(**config['model'])
            freeze_kws = config.get('freeze_keywords', None)
            freeze_excl_kws = config.get('freeze_exclusive_keywords', None)
            if freeze_kws:
                self.logger.info(f'Freezing parameters that contain keywords '
                                 f'`{freeze_kws}` in model `{name}`.',
                                 indent_level=2)
                Freezer.freeze_by_keywords(self.models[name],
                                           keywords=freeze_kws,
                                           exclusive_keywords=freeze_excl_kws)
                if len(list(self.models[name].parameters())) == 0:
                    self.logger.warning(f'Freezing the entire model `{name}` '
                                        f'omits the optimizer and the learning '
                                        f'rate scheduler, which may require a '
                                        f'redesign of both `Runner` and '
                                        f'`Loss`.', indent_level=2)
                    config.pop('lr', None)
                    config.pop('opt', None)
            self.models[name].eval().requires_grad_(False).cuda()
            self.model_kwargs_init[name] = config['model']
            self.model_kwargs_train[name] = config.get('kwargs_train', dict())
            self.model_kwargs_val[name] = config.get('kwargs_val', dict())
            self.model_has_unused_param[name] = config.get(
                'has_unused_parameters', False)
            self.model_broadcast_buffers[name] = config.get(
                'broadcast_buffers', True)
            self.opt_config[name] = config.get('opt', None)
            self.lr_config[name] = config.get('lr', None)

            if self.model_kwargs_train[name]:
                self.logger.info('Training kwargs:', indent_level=2)
                for key, val in self.model_kwargs_train[name].items():
                    self.logger.info(f'{key}: {val}', indent_level=3)
            else:
                self.logger.info('Training kwargs: {}', indent_level=2)
            if self.model_kwargs_val[name]:
                self.logger.info('Validation kwargs:', indent_level=2)
                for key, val in self.model_kwargs_val[name].items():
                    self.logger.info(f'{key}: {val}', indent_level=3)
            else:
                self.logger.info('Validation kwargs: {}', indent_level=2)
            self.logger.info(f'Finish building model `{name}`.', indent_level=1)
        self.logger.info('Finish building models.\n')

    def log_model_info(self):
        """Logs model related information."""
        self.logger.info('Model information:')

        model_info = ''
        for name, model in self.models.items():
            model_info += ('\n' + '=' * 50 + '\n')

            model_info += f'Name: {name}\n'
            model_info += ('-' * 50 + '\n')

            model_info += 'Structure:\n\n'
            model_info += str(model) + '\n'
            model_info += ('-' * 50 + '\n')

            # Statistics headers.
            name_header = 'Name'
            name_separator = '-' * len(name_header)
            shape_header = 'Shape'
            shape_separator = '-' * len(shape_header)
            numel_header = '# Params'
            numel_separator = '-' * len(numel_header)

            model_info += 'Parameters:\n\n'
            param_shapes = dict()
            param_numels = dict()
            for param_name, param in model.named_parameters():
                param_shapes[param_name] = f'{list(param.shape)}'
                param_numels[param_name] = param.numel()
            if len(param_shapes) == 0:  # no parameters
                model_info += 'The model contains no parameter.\n'
            else:
                param_name_max_len = max(map(len, param_shapes.keys()))
                param_name_max_len = max(param_name_max_len, len(name_header))
                param_shape_max_len = max(map(len, param_shapes.values()))
                param_shape_max_len = max(param_shape_max_len,
                                          len(shape_header))
                param_numel_max_len = int(np.ceil(
                    np.log10(max(param_numels.values()))
                ))
                param_numel_max_len = max(param_numel_max_len,
                                          len(numel_header))
                model_info += f'{name_header:<{param_name_max_len + 2}}'
                model_info += f'{shape_header:<{param_shape_max_len + 2}}'
                model_info += f'{numel_header:>{param_numel_max_len + 2}}\n'
                model_info += f'{name_separator:<{param_name_max_len + 2}}'
                model_info += f'{shape_separator:<{param_shape_max_len + 2}}'
                model_info += f'{numel_separator:>{param_numel_max_len + 2}}\n'
                for param_name, param_shape in param_shapes.items():
                    param_numel = param_numels[param_name]
                    model_info += f'{param_name:<{param_name_max_len + 2}}'
                    model_info += f'{param_shape:<{param_shape_max_len + 2}}'
                    model_info += f'{param_numel:{param_numel_max_len + 2}d}\n'
            model_info += ('-' * 50 + '\n')

            model_info += 'Buffers:\n\n'
            buffer_shapes = dict()
            buffer_numels = dict()
            for buffer_name, buffer in model.named_buffers():
                buffer_shapes[buffer_name] = f'{list(buffer.shape)}'
                buffer_numels[buffer_name] = buffer.numel()
            if len(buffer_shapes) == 0:  # no buffers
                model_info += 'The model contains no buffer.\n'
            else:
                buffer_name_max_len = max(map(len, buffer_shapes.keys()))
                buffer_name_max_len = max(buffer_name_max_len, len(name_header))
                buffer_shape_max_len = max(map(len, buffer_shapes.values()))
                buffer_shape_max_len = max(buffer_shape_max_len,
                                           len(shape_header))
                buffer_numel_max_len = int(np.ceil(
                    np.log10(max(buffer_numels.values()))
                ))
                buffer_numel_max_len = max(buffer_numel_max_len,
                                           len(numel_header))
                model_info += f'{name_header:<{buffer_name_max_len + 2}}'
                model_info += f'{shape_header:<{buffer_shape_max_len + 2}}'
                model_info += f'{numel_header:>{buffer_numel_max_len + 2}}\n'
                model_info += f'{name_separator:<{buffer_name_max_len + 2}}'
                model_info += f'{shape_separator:<{buffer_shape_max_len + 2}}'
                model_info += f'{numel_separator:>{buffer_numel_max_len + 2}}\n'
                for buffer_name, buffer_shape in buffer_shapes.items():
                    buffer_numel = buffer_numels[buffer_name]
                    model_info += f'{buffer_name:<{buffer_name_max_len + 2}}'
                    model_info += f'{buffer_shape:<{buffer_shape_max_len + 2}}'
                    model_info += f'{buffer_numel:{buffer_numel_max_len + 2}d}'
                    model_info += '\n'
            model_info += ('-' * 50 + '\n')

            model_info += 'Size (using `Float32` for size computation):\n\n'
            param_size = sum(param_numels.values())
            buffer_size = sum(buffer_numels.values())
            total_size = param_size + buffer_size
            model_info += 'Parameters: '
            model_info += f'{param_size:10d} params '
            model_info += f'{param_size * 4:10d} bytes '
            model_info += f'({param_size * 4 / 1024 / 1024:7.2f} MB)\n'
            model_info += 'Buffers:    '
            model_info += f'{buffer_size:10d} params '
            model_info += f'{buffer_size * 4:10d} bytes '
            model_info += f'({buffer_size * 4 / 1024 / 1024:7.2f} MB)\n'
            model_info += ('-' * 10 + '\n')
            model_info += 'Total:      '
            model_info += f'{total_size:10d} params '
            model_info += f'{total_size * 4:10d} bytes '
            model_info += f'({total_size * 4/ 1024 / 1024:7.2f} MB)\n'
            model_info += ('=' * 50 + '\n')

        self.logger.print(model_info)

    def distribute(self):
        """Distributes model across multiple replicas."""
        self.logger.info(f'Setting up distributed models across '
                         f'{self.world_size} GPU(s) ...')
        for name, model in self.models.items():
            # DDP is only applicable to a model with parameters.
            # Hence, an entire frozen model will bypass the wrapping of DDP
            # after its buffers got synchronized across replicas.
            if len(list(model.parameters())) == 0:
                self.logger.warning(f'No parameters contained in model '
                                    f'`{name}`, hence DDP model is the same as '
                                    f'native PyTorch model.')
                # Synchronize all buffers.
                for buffer in model.buffers():
                    dist.broadcast(buffer, src=0)
                self.ddp_models[name] = model
                continue

            model_info = f'Model `{name}`'

            # Trainable.
            if name in self.opt_config and self.opt_config[name] is not None:
                if self.model_has_unused_param[name]:
                    model_info += ' has unused parameters,'
                else:
                    model_info += ' does not have unused parameters,'
                if self.model_broadcast_buffers[name]:
                    model_info += ' and broadcasts buffers.'
                else:
                    model_info += ' and does not broadcast buffers.'
                model.train().requires_grad_(True)
                self.ddp_models[name] = DDP(
                    module=model,
                    device_ids=[self.device],
                    broadcast_buffers=self.model_broadcast_buffers[name],
                    find_unused_parameters=self.model_has_unused_param[name])

            # Not trainable.
            else:
                model_info += ' is not trainable, hence, not distributed.'
                # `DDP` is called ONLY for synchronizing parameters at the
                # beginning.
                model.requires_grad_(True)
                _ = DDP(
                    module=model,
                    device_ids=[self.device],
                    broadcast_buffers=True,
                    find_unused_parameters=False)
                model.eval().requires_grad_(False)

            self.logger.info(model_info, indent_level=1)
        self.logger.info('Finish setting up distributed models.\n')

    def build_optimizers(self):
        """Builds optimizers for models if needed."""
        if len(self.opt_config) == 0:
            return

        self.logger.info('Building optimizers ...')
        for name, config in self.opt_config.items():
            if not config:
                continue
            if name in self.optimizers:
                raise ValueError(f'Optimizer `{name}` has already existed!')
            if name not in self.models:
                raise ValueError(f'Model `{name}` is missing!')
            self.optimizers[name] = build_optimizer(config, self.models[name])
            self.logger.info(f'Model `{name}`:', indent_level=1)
            for key, val in config.items():
                self.logger.info(f'{key}: {val}', indent_level=2)
        self.logger.info('Finish building optimizers.\n')

    def build_lr_scheduler(self):
        """Builds learning rate scheduler as a controller if needed."""
        if len(self.lr_config) == 0:
            return

        self.logger.info('Building learning rate scheduler ...')
        self.controllers.append(build_controller('LRScheduler', self.lr_config))
        self.logger.info('Finish building learning rate scheduler.\n')

    def build_loss(self):
        """Builds loss functions."""
        if not self.config.loss:
            return

        self.logger.info('Building loss function ...')
        self.loss = build_loss(self, **self.config.loss)
        self.logger.info('Finish building loss function.\n')

    def build_augment(self):
        """Builds augmentation pipeline."""
        self.logger.info('Building differentiable augmentation pipeline ...')
        self.augment = build_aug(**self.config.aug)
        self.augment.train().requires_grad_(False).cuda()
        self.augment_kwargs = self.config.aug_kwargs or dict()
        self.logger.info('Augmentation settings:', indent_level=1)
        for key, val in self.config.aug.items():
            self.logger.info(f'{key}: {val}', indent_level=2)
        if self.augment_kwargs:
            self.logger.info('Augmentation runtime kwargs:', indent_level=1)
            for key, val in self.augment_kwargs.items():
                self.logger.info(f'{key}: {val}', indent_level=2)
        else:
            self.logger.info('Augmentation runtime kwargs: {}', indent_level=1)
        self.logger.info('Finish building augmentation pipeline.\n')

        """Builds augmentation pipeline."""
        self.logger.info('Building object differentiable augmentation pipeline ...')
        self.object_augment = build_aug(**self.config.object_aug)
        self.object_augment.train().requires_grad_(False).cuda()
        self.object_augment_kwargs = self.config.object_aug_kwargs or dict()
        self.logger.info('Object Augmentation settings:', indent_level=1)
        for key, val in self.config.object_aug.items():
            self.logger.info(f'{key}: {val}', indent_level=2)
        if self.object_augment_kwargs:
            self.logger.info('Object Augmentation runtime kwargs:', indent_level=1)
            for key, val in self.object_augment_kwargs.items():
                self.logger.info(f'{key}: {val}', indent_level=2)
        else:
            self.logger.info('Augmentation runtime kwargs: {}', indent_level=1)
        self.logger.info('Finish building object augmentation pipeline.\n')


    def build_controllers(self):
        """Builds timer and additional controllers besides LRScheduler."""
        self.logger.info('Building controllers ...')
        self.timer = build_controller('Timer')
        for ctrl_type, ctrl_config in self.config.controllers.items():
            self.controllers.append(build_controller(ctrl_type, ctrl_config))
        self.controllers.sort(key=lambda x: x.priority)
        for controller in self.controllers:
            self.logger.info(f'{controller.name}', indent_level=1)
            controller.start(self)
        self.logger.info('Finish building controllers.\n')

    def build_metrics(self):
        """Builds metrics used for model evaluation."""
        self.logger.info('Building metrics ...')
        for metric_type, metric_config in self.config.metrics.items():
            metric = dict()

            # Settings for metric computation.
            init_kwargs = metric_config.get('init_kwargs', dict())
            init_kwargs['work_dir'] = self.result_dir
            init_kwargs['logger'] = self.logger
            init_kwargs['tb_writer'] = self.tb_writer
            if 'batch_size' not in init_kwargs:
                init_kwargs['batch_size'] = self.val_batch_size
            metric['fn'] = build_metric(metric_type, **init_kwargs)

            # Evaluation kwargs should be a dictionary, where each key stands
            # for a model name in `self.models`, specifying the model to test,
            # while each value contains the runtime kwargs for model forward,
            # specifying the model behavior when testing.
            eval_kwargs = metric_config.get('eval_kwargs', dict())
            for key, val in eval_kwargs.items():
                assert isinstance(key, str)
                assert isinstance(val, dict)
            metric['kwargs'] = eval_kwargs

            # Evaluation interval.
            metric['interval'] = metric_config.get('interval', None)
            metric['first_iter'] = metric_config.get('first_iter', None)
            interval = metric['interval']
            first_iter = metric['first_iter']

            # Settings for saving best checkpoint.
            metric['save_best'] = metric_config.get('save_best', None)
            metric['save_running_metadata'] = metric_config.get(
                'save_running_metadata', None)
            metric['save_optimizer'] = metric_config.get('save_optimizer', None)
            metric['save_learning_rate'] = metric_config.get(
                'save_learning_rate', None)
            metric['save_loss'] = metric_config.get('save_loss', None)
            metric['save_augment'] = metric_config.get('save_augment', None)
            metric['save_running_stats'] = metric_config.get(
                'save_running_stats', None)
            save_best = metric['save_best']
            save_running_metadata = metric['save_running_metadata']
            save_optimizer = metric['save_optimizer']
            save_learning_rate = metric['save_learning_rate']
            save_loss = metric['save_loss']
            save_augment = metric['save_augment']
            save_running_stats = metric['save_running_stats']

            self.metrics[metric['fn'].name] = metric

            self.logger.info(metric_type, indent_level=1)
            for key, val in metric['fn'].info().items():
                self.logger.info(f'{key}: {val}', indent_level=2)
            self.logger.info(f'Evaluation interval: {interval}', indent_level=2)
            self.logger.info(f'Evaluate on first iter: {first_iter}',
                             indent_level=2)
            if save_best:
                self.logger.info('Save the best checkpoint:', indent_level=2)
                self.logger.info(
                    f'Saving running metadata: {save_running_metadata}',
                    indent_level=3)
                self.logger.info(f'Saving optimizer state: {save_optimizer}',
                                 indent_level=3)
                self.logger.info(
                    f'Saving learning rate scheduler: {save_learning_rate}',
                    indent_level=3)
                self.logger.info(f'Saving loss: {save_loss}', indent_level=3)
                self.logger.info(f'Saving augment: {save_augment}',
                                 indent_level=3)
                self.logger.info(f'Saving running stats: {save_running_stats}',
                                 indent_level=3)
            else:
                self.logger.info('Do not save the best checkpoint.',
                                 indent_level=2)
        self.logger.info('Finish building metrics.\n')

    def build_train_loader(self):
        """Builds training data loader."""
        self.logger.info('Building `train` data loader ...')
        self.train_loader = build_dataset(
            for_training=True,
            batch_size=self.batch_size,
            dataset_kwargs=self.config.data.train,
            data_loader_kwargs=self.config.data.loader.copy())
        self.log_data_info(self.train_loader)
        self.logger.info('Finish building `train` data loader.\n')

    def build_val_loader(self):
        """Builds validation data loader."""
        self.logger.info('Building `val` data loader ...')
        self.val_loader = build_dataset(
            for_training=False,
            batch_size=self.val_batch_size,
            dataset_kwargs=self.config.data.val,
            data_loader_kwargs=self.config.data.loader.copy())
        self.log_data_info(self.val_loader)
        self.logger.info('Finish building `val` data loader.\n')

    def log_data_info(self, data_loader):
        """Logs data related information."""
        # Log dataset info.
        self.logger.info('Dataset information:', indent_level=1)
        for key, val in data_loader.dataset.info().items():
            self.logger.info(f'{key}: {val}', indent_level=2)
        # Log sampler info.
        self.logger.info('Sampler information:', indent_level=1)
        for key, val in data_loader.sampler.info().items():
            self.logger.info(f'{key}: {val}', indent_level=2)
        # Log data loader info.
        self.logger.info('Data loader information:', indent_level=1)
        for key, val in data_loader.info().items():
            self.logger.info(f'{key}: {val}', indent_level=2)

    def clip_model_gradient(self, name, nan=0.0, min_val=-1e5, max_val=1e5):
        """Clips the gradient of a particular model.

        Args:
            name: The name of the model, i.e., in `self.models`.
            nan: The value to which `nan` is clipped. This should always be set
                as `0`. (default: 0)
            min_val: The minimum value to cutoff. (default: -1e5)
            max_val: The maximum value to cutoff. (default: 1e5)
        """
        assert nan == 0
        for param_name, param in self.models[name].named_parameters():
            if param.grad is None:
                if self.model_has_unused_param[name]:
                    continue
                raise ValueError(f'Parameter `{param_name}` from '
                                 f'model `{name}` does not have gradient!')
            if min_val is None:
                min_val = torch.finfo(param.grad.dtype).min
            if max_val is None:
                max_val = torch.finfo(param.grad.dtype).max
            torch.clamp(param.grad.unsqueeze(0).nansum(0),
                        min=min_val, max=max_val, out=param.grad)

    def zero_grad_optimizer(self, name, set_to_none=None):
        """Wraps `optimizer.zero_grad()` with `set_to_none` option.

        When clear gradients, setting `set_to_none` as `True` is slightly
        efficient, however, it may cause the problem of `adding tensor with
        None` when some gradients are missing. By default, we use
        `has_unused_parameter` to determine whether the gradient should be set
        to zeros or None.
        """
        if set_to_none is None:
            set_to_none = not self.model_has_unused_param[name]
        self.optimizers[name].zero_grad(set_to_none=set_to_none)

    def unscale_optimizer(self, name):
        self.amp_scaler.unscale_(self.optimizers[name])
    def step_optimizer(self, name, clip_grad=True, **clip_kwargs):
        """Wraps stepping optimizer with gradient clip and AMP scalar."""
        # NOTE: AMP will use inf/NaN to adjust its behavior, hence the gradient
        # should not be clipped.
        if not self.enable_amp and clip_grad:
            self.clip_model_gradient(name, **clip_kwargs)
        self.amp_scaler.step(self.optimizers[name])

    @staticmethod
    def smooth_model(src, avg, beta=0.999):
        """Smooths model weights with moving average.

        This trick is commonly used in GAN training, where the weight of the
        generator is life-long averaged.

        NOTE: `src` and `avg` are assumed to be with exactly the same structure.

        Args:
            src: The source model used to update the averaged weights.
            avg: The averaged model weights.
            beta: Hyper-parameter used for moving average. (default: 0.999)
        """
        with torch.no_grad(), ddp_sync(src, False), ddp_sync(avg, False):
            # Update parameters with moving average.
            for src_p, avg_p in zip(src.parameters(), avg.parameters()):
                avg_p.copy_(src_p.lerp(avg_p, beta))
            # Directly copy buffers.
            for src_b, avg_b in zip(src.buffers(), avg.buffers()):
                avg_b.copy_(src_b)

    def pre_execute_controllers(self):
        """Pre-executes all controllers in order of priority."""
        for controller in self.controllers:
            controller.pre_execute(self)

    def post_execute_controllers(self):
        """Post-executes all controllers in order of priority."""
        for controller in self.controllers:
            controller.post_execute(self)

    def start(self):
        """Starts runner by starting timer."""
        self.timer.start(self)
        dist.barrier()  # Start all replicas together.

    def finish(self):
        """Finishes runner by ending controllers and timer."""
        for controller in self.controllers:
            controller.end(self)
        if self.tb_writer is not None:
            self.tb_writer.close()
        self.timer.end(self)
        dist.barrier()  # Make sure all replicas finish.

    def train_step(self, data):
        """Executes one training step."""
        raise NotImplementedError('Should be implemented in derived class!')

    def train(self):
        """Training function."""
        self.logger.info('Start training.\n')
        self.start()

        with Profiler(enable=self.config.enable_profiler,
                      tb_dir=self.profile_dir,
                      logger=self.logger,
                      **self.config.profiler_schedule_kwargs) as profiler:
            while self.iter < self.total_iters:
                self._iter += 1

                # Pre-execute all controllers before each training step.
                self.pre_execute_controllers()

                # Fetch a batch of samples.
                batch_data = next(self.train_loader)
                if not isinstance(batch_data, dict):
                    batch_data = {'data': batch_data}

                # Start timer before each training step. (Computing `data_time`)
                self.timer.pre_execute(self)

                # Move data to GPU if needed.
                for key in batch_data:
                    if not isinstance(batch_data[key], torch.Tensor):
                        continue
                    assert batch_data[key].shape[0] == self.batch_size
                    batch_data[key] = batch_data[key].cuda()

                # Execute training step.
                self.batch_data = batch_data  # For viz ONLY.
                self.train_step(batch_data)

                # Update basic information.
                self.seen_img += self.minibatch
                if self.enable_amp:
                    self.running_stats.update(
                        {'Misc/AMP Scale': self.amp_scaler.get_scale()})

                # End timer after each training step. (Computing `iter_time`)
                self.timer.post_execute(self)

                # Post-execute all controllers after each training step.
                self.post_execute_controllers()

                # Update profiler.
                profiler.step()

        self.finish()
        self.logger.print()
        self.logger.info(f'Finish training job `{self.job_name}` in '
                         f'{format_time(self.end_time - self.start_time)}.')
        for metric_name, results in self.eval_results.items():
            if 'best' not in results:
                continue
            best_result, best_iter = results['best']
            self.logger.info(f'Best `{metric_name}`: {best_result} at iter '
                             f'{best_iter:06d}.', indent_level=1)

    def check_ddp_consistency(self):
        """Checks model consistency across multiple replicas.

        This function is used to make sure that all replicas have the same
        parameters.
        """
        for model_name, model in self.models.items():
            for param_name, param in model.named_parameters():
                src = param.detach()
                other = src.clone()
                dist.broadcast(tensor=other, src=0)  # Broadcast 0 to others.
                if not (src == other).all():
                    raise SystemExit(f'Parameter `{param_name}` from '
                                     f'model `{model_name}` mismatch between '
                                     f'rank 0 and rank {self.rank}.')

    def save(self,
             filepath,
             running_metadata=True,
             optimizer=True,
             learning_rate=True,
             loss=True,
             augment=True,
             running_stats=False):
        """Saves the current running status.
        Args:
            filepath: File path to save the checkpoint.
            running_metadata: Whether to save the running metadata, such as
                batch size, current iteration, etc. (default: True)
            optimizer: Whether to save the optimizer. (default: True)
            learning_rate: Whether to save the learning rate. (default: True)
            loss: Whether to save the loss. (default: True)
            augment: Whether to save the augmentation, especially the adaptive
                augmentation probability. (default: True)
            running_stats: Whether to save the running stats. (default: False)
        """
        if os.path.isfile(filepath):
            self.logger.warning(f'{filepath} already exists, this will '
                                f'overwrite the previous one.')
        checkpoint = dict()
        # Models.
        checkpoint['models'] = dict()
        for name, model in self.models.items():
            checkpoint['models'][name] = model.state_dict()
        checkpoint['model_kwargs_init'] = self.model_kwargs_init
        checkpoint['model_kwargs_train'] = self.model_kwargs_train
        checkpoint['model_kwargs_val'] = self.model_kwargs_val
        checkpoint['model_has_unused_param'] = self.model_has_unused_param
        checkpoint['model_broadcast_buffers'] = self.model_broadcast_buffers
        # Running metadata.
        if running_metadata:
            checkpoint['running_metadata'] = {
                'iter': self.iter,
                'seen_img': self.seen_img,
            }
        # Optimizers.
        if optimizer:
            checkpoint['optimizers'] = dict()
            for opt_name, opt in self.optimizers.items():
                checkpoint['optimizers'][opt_name] = opt.state_dict()
            checkpoint['opt_config'] = self.opt_config
        # Learning rates.
        if learning_rate:
            checkpoint['learning_rates'] = dict()
            for lr_name, lr in self.lr_schedulers.items():
                checkpoint['learning_rates'][lr_name] = lr.state_dict()
            checkpoint['lr_config'] = self.lr_config
        # Loss.
        if loss:
            checkpoint['loss'] = self.loss.state_dict()
        # Augmentation.
        if augment:
            checkpoint['augment'] = self.augment.state_dict()
        # Running stats (only save `stats_pool`).
        if running_stats:
            checkpoint['running_stats'] = self.running_stats.stats_pool
        # Save checkpoint.
        torch.save(checkpoint, filepath)
        self.logger.info(f'Successfully saved checkpoint to `{filepath}`.')

    def load(self,
             filepath,
             running_metadata=True,
             optimizer=True,
             learning_rate=True,
             loss=True,
             augment=True,
             running_stats=False,
             map_location='cpu'):
        """Loads previous running status.

        Args:
            filepath: File path to load the checkpoint.
            running_metadata: Whether to load the running metadata, such as
                batch size, current iteration, etc. (default: True)
            optimizer: Whether to load the optimizer. (default: True)
            learning_rate: Whether to load the learning rate. (default: True)
            loss: Whether to load the loss. (default: True)
            augment: Whether to load the augmentation, especially the adaptive
                augmentation probability. (default: True)
            running_stats: Whether to load the running stats. (default: False)
            map_location: Map location used for model loading. (default: `cpu`)
        """
        self.logger.info(f'Resuming from checkpoint `{filepath}` ...')
        if not os.path.isfile(filepath):
            raise IOError(f'Checkpoint `{filepath}` does not exist!')
        map_location = map_location.lower()
        assert map_location in ['cpu', 'gpu']
        if map_location == 'gpu':
            map_location = lambda storage, location: storage.cuda(self.device)
        checkpoint = torch.load(filepath, map_location=map_location)
        # Load models.
        if 'models' not in checkpoint:
            checkpoint = {'models': checkpoint}
        for model_name, model in self.models.items():
            if model_name not in checkpoint['models']:
                self.logger.warning(f'Model `{model_name}` is not included in '
                                    f'the checkpoint, and hence will NOT be '
                                    f'loaded!', indent_level=1)
                continue
            state_dict = checkpoint['models'][model_name]
            model.load_state_dict(state_dict)
            self.logger.info(f'Successfully loaded model `{model_name}`.',
                             indent_level=1)
        # Load running metadata.
        if running_metadata:
            if 'running_metadata' not in checkpoint:
                self.logger.warning('Running metadata is not included in the '
                                    'checkpoint, and hence will NOT be '
                                    'loaded!', indent_level=1)
            else:
                self._iter = checkpoint['running_metadata']['iter']
                self._start_iter = self._iter
                self.seen_img = checkpoint['running_metadata']['seen_img']
        # Load optimizers.
        if optimizer:
            if 'optimizers' not in checkpoint:
                self.logger.warning('Optimizers are not included in the '
                                    'checkpoint, and hence will NOT be '
                                    'loaded!', indent_level=1)
            else:
                for opt_name, opt in self.optimizers.items():
                    if opt_name not in checkpoint['optimizers']:
                        self.logger.warning(f'Optimizer `{opt_name}` is not '
                                            f'included in the checkpoint, and '
                                            f'hence will NOT be loaded!',
                                            indent_level=1)
                        continue
                    opt.load_state_dict(checkpoint['optimizers'][opt_name])
                    self.logger.info(f'Successfully loaded optimizer '
                                     f'`{opt_name}`.', indent_level=1)
        # Load learning rates.
        if learning_rate:
            if 'learning_rates' not in checkpoint:
                self.logger.warning('Learning rates are not included in the '
                                    'checkpoint, and hence will NOT be '
                                    'loaded!', indent_level=1)
            else:
                for lr_name, lr in self.lr_schedulers.items():
                    if lr_name not in checkpoint['learning_rates']:
                        self.logger.warning(f'Learning rate `{lr_name}` is not '
                                            f'included in the checkpoint, and '
                                            f'hence will NOT be loaded!',
                                            indent_level=1)
                        continue
                    lr.load_state_dict(checkpoint['learning_rates'][lr_name])
                    self.logger.info(f'Successfully loaded learning rate '
                                     f'`{lr_name}`.', indent_level=1)
        # Load loss.
        if loss:
            if 'loss' not in checkpoint:
                self.logger.warning('Loss is not included in the checkpoint, '
                                    'and hence will NOT be loaded!',
                                    indent_level=1)
            else:
                self.loss.load_state_dict(checkpoint['loss'])
                self.logger.info('Successfully loaded loss.', indent_level=1)
        # Load augmentation.
        if augment:
            if 'augment' not in checkpoint:
                self.logger.warning('Augmentation is not included in '
                                    'the checkpoint, and hence will NOT be '
                                    'loaded!', indent_level=1)
            else:
                self.augment.load_state_dict(checkpoint['augment'])
                self.logger.info('Successfully loaded augmentation.',
                                 indent_level=1)
        # Load running stats.
        #  Only resume `stats_pool` from checkpoint.
        if running_stats:
            if 'running_stats' not in checkpoint:
                self.logger.warning('Running stats is not included in the '
                                    'checkpoint, and hence will NOT be '
                                    'loaded!', indent_level=1)
            else:
                self.running_stats.stats_pool = deepcopy(
                    checkpoint['running_stats'])
                self.running_stats.is_resumed = True  # avoid conflicts when add
                self.logger.info('Successfully loaded running stats.',
                                 indent_level=1)
        # Log message.
        tailing_message = ''
        if running_metadata and 'running_metadata' in checkpoint:
            tailing_message = f' (iteration {self.iter})'
        self.logger.info(f'Successfully loaded from checkpoint `{filepath}`.'
                         f'{tailing_message}')
