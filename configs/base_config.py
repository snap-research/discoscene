# python3.7
"""Contains the basic configurations that are applicable to all tasks.

Each task will be assigned a working directory, such that (1) the data and
pre-trained models on which the training is relied will be linked/downloaded to
such a directory, (2) the intermediate results, checkpoints, log, etc. will be
saved to such a directory, and (3) the TensorBoard events, performance profile,
and computing resources required by the training will be saved to such a
directory. Basically, the working directory is organized as follows:

/${ROOT_WORK_DIR}/
└── ${JOB_NAME}/
    ├── ${DATA_DIR}/
    ├── ${CHECKPOINT_DIR}/
    ├── ${RESULT_DIR}/
    ├── ${TENSORBOARD_DIR}/
    ├── ${PROFILE_DIR}/
    ├── ${RESOURCE_DIR}/
    ├── ${CONFIG_FILENAME}  # in JSON format
    ├── ${LOG_DATA_FILENAME}  # in JSON Lines format
    └── ${LOG_FILENAME}  # in plain text
"""

import os.path
import sys
from collections import defaultdict
from easydict import EasyDict
import cloup

from utils.parsing_utils import parse_json
from utils.parsing_utils import IntegerParamType
from utils.parsing_utils import FloatParamType
from utils.parsing_utils import BooleanParamType
from utils.parsing_utils import JsonParamType
from utils.parsing_utils import IndexParamType

__all__ = ['BaseConfig']

_PARAM_TYPE_TO_VALUE_TYPE = {
    'IntegerParamType': 'int',
    'FloatParamType': 'float',
    'BooleanParamType': 'bool',
    'StringParamType': 'str',
    'IndexParamType': 'index-string',
    'JsonParamType': 'json-string'
}


class BaseConfig(object):
    """Defines the base configuration class.

    `cloup` (https://pypi.org/project/cloup/), which is primarily based on
    `click` (https://palletsprojects.com/p/click/), is used for user
    interaction. Basically, each task corresponds to a command, consisting of
    multiple options.

    The base class provides the following static variables, which MUST be
    overridden by a derived class:

    (1) name: Name of the configuration, which will be used as the command name.
    (2) hint: A brief description of the configuration, which will be used as
        the hint (short help message) of the command.
    (3) info: A detailed description of the configuration, which will be used as
        the help message of the command.

    The base class wraps `click.option` with `cls.command_option`, and also
    provides the following option types beyond `click`:

    (1) int_type: The input string will be parsed as an integer. Different from
        the conventional `int` type, this type supports parsing string `null`
        and `none` as `None`. Please refer to `utils/parsing_utils.py` for more
        details.
    (2) float_type: The input string will be parsed as a float. Different from
        the conventional `float` type, this type supports parsing string `null`
        and `none` as `None`. Please refer to `utils/parsing_utils.py` for more
        details.
    (3) bool_type: The input string will be parsed as a boolean. Please refer to
        `utils/parsing_utils.py` for more details.
    (4) index_type: The input string will be parsed as a list of integers as
        indices. Please refer to `utils/parsing_utils.py` for more details.
    (5) json_type: The input string will be parsed following JSON format. Please
        refer to `utils/parsing_utils.py` for more details.

    The base class provides the following functions to parse configuration from
    command line:

    - Functions requiring implementation in derived class:

    (1) get_options(): Declare all options required by a particular task. The
        base class has already pre-declared some options that will be shared
        across tasks (e.g., data-related options). To declare more options,
        the derived class should override this function by first calling
        `options = super().get_options()`.
    (2) parse_options(): Parse the options obtained from the command line (as
        well as those options with default values) to `self.config`. This is the
        core function of the configuration class, which converts `options` to
        `configurations`.
    (3) get_recommended_options(): Get a list of options that are recommended
        for a particular task. The base class has already pre-declared some
        recommended options that will be shared across tasks. To recommend more
        options, the derived class should override this function by first
        calling `recommended_opts = super().get_recommended_options()`.

    - Helper functions shared by all derived classes:

    (1) inspect_option(): Inspect argument from a particular `click.option`,
        including the argument name, argument type, default value, and help
        message.
    (2) add_options_to_command(): Add all options for a particular task to the
        corresponding command. This function is specially designed to show
        user-friendly help message.
    (3) get_command(): Return a `click.command` to get interactive with users.
        This function makes it possible to pass options through command line.
    (4) update_config(): Update the configuration parsed from options with
        key-value pairs. This function makes option parsing more flexible.
    (5) get_config(): The main function to get the parsed configuration, which
        wraps functions `parse_options()` and `update_config()`.

    In summary, to define a configuration class for a new task, the derived
    class only need to implement `get_options()` to declare changeable settings
    as well as the default values, and `parse_options()` to parse the settings
    to `self.config`.
    """

    name = None  # Configuration name, which will be used as the command name.
    hint = None  # Command hint, which is a brief description.
    info = None  # Command help message, which is a detailed description.

    int_type = IntegerParamType()
    float_type = FloatParamType()
    bool_type = BooleanParamType()
    index_type = IndexParamType()
    json_type = JsonParamType()
    command_option = cloup.option

    @staticmethod
    def inspect_option(option):
        """Inspects argument from a particular option.

        Args:
            option: The input `click.option` to inspect.

        Returns:
            An `EasyDict` indicating the `name`, `type`, `default` (default
                value), and `help` (help message) of the argument.
        """

        @option
        def func():
            """A dummy function used to parse decorator."""

        arg = func.__click_params__[0]
        return EasyDict(
            name=arg.name,
            type=_PARAM_TYPE_TO_VALUE_TYPE[arg.type.__class__.__name__],
            default=arg.default,
            help=arg.help
        )

    @staticmethod
    def add_options_to_command(options):
        """Adds task options to a command.

        This function can be used as a decorator, which wraps a command with
        a collection of options. This function also supports grouping options to
        make the help message more user-friendly.

        Args:
            options: A dictionary, with each key as a group name and the
                corresponding value is a list of options under such a group.
                Usually, this field is provided by `cls.get_options()`.
        """
        def _add_options(func):
            for opt_group, opts in reversed(list(options.items())):
                func = cloup.option_group(opt_group, *opts)(func)
            return func
        return _add_options

    @classmethod
    def get_command(cls):
        """Gets a `click.command` for this class, containing all options.

        The static variables `name`, `hint`, and `info` will be used as the
        name, short help message, and help message of the command, respectively.
        """
        assert cls.name, 'Configuration name is required!'
        assert cls.hint, 'Brief configuration description is required!'
        assert cls.info, 'Detailed configuration description is required!'

        @cloup.command(name=cls.name, short_help=cls.hint, help=cls.info)
        @cls.add_options_to_command(cls.get_options())
        @cls.command_option(
            '--options', '-o', type=str, multiple=True,
            help='Please use `-o key_1=val_1 -o key_2=val_2` to flexibly '
                 'update the configuration. `key`s are expected to be with '
                 'hierarchical format, e.g., `data.train.root_dir`, while '
                 '`val`s are expected to be with JSON format.')
        def _command(**kwargs):
            return kwargs
        return _command

    @classmethod
    def get_options(cls):
        """Declares options for the configuration."""
        options = defaultdict(list)

        options['Working directory settings'].extend([
            cls.command_option(
                '--root_work_dir', type=str, default='work_dirs',
                help='Root to all working directories.'),
            cls.command_option(
                '--job_name', type=str, default=cls.name,
                help='Job name.'),
            cls.command_option(
                '--data_dir', type=str, default='data',
                help='From which to read data, pre-trained weights, etc.'),
            cls.command_option(
                '--checkpoint_dir', type=str, default='checkpoints',
                help='To which to save checkpoints.'),
            cls.command_option(
                '--result_dir', type=str, default='results',
                help='To which to save intermediate results, metrics, etc.'),
            cls.command_option(
                '--tensorboard_dir', type=str, default='events',
                help='To which to save TensorBoard events.'),
            cls.command_option(
                '--profile_dir', type=str, default='profile',
                help='To which to save performance profile.'),
            cls.command_option(
                '--resource_dir', type=str, default='resources',
                help='To which to save computing resources used.'),
            cls.command_option(
                '--config_path', type=str, default='config.json',
                help='To which to save full configuration.'),
            cls.command_option(
                '--log_data_path', type=str, default='log.jsonl',
                help='To which to save raw log data, e.g. losses.'),
            cls.command_option(
                '--log_path', type=str, default='log.txt',
                help='To which to save full log.'),
            cls.command_option(
                '--interactive', type=cls.bool_type, default=False,
                help='Whether is a interactive job')
        ])

        options['Logging and file system settings'].extend([
            cls.command_option(
                '--logger_type', type=str, default='rich',
                help='Type of logger.'),
            cls.command_option(
                '--use_tensorboard', type=cls.bool_type, default=True,
                help='Whether to use TensorBoard for log visualization.'),
            cls.command_option(
                '--file_transmitter_type', type=str, default='dummy',
                help='Type of file transmitter.'),
            cls.command_option(
                '--file_transmitter_kwargs', type=cls.json_type, default=None,
                help='Additional keyword arguments for file transmitter.')
        ])

        options['CUDNN settings'].extend([
            cls.command_option(
                '--enable_cudnn', type=cls.bool_type, default=True,
                help='Switch this on to enable CUDNN acceleration.'),
            cls.command_option(
                '--cudnn_benchmark', type=cls.bool_type, default=True,
                help='Switching this on can speed up the computation of static '
                     'graph, but may harm the reproducibility.'),
            cls.command_option(
                '--cudnn_deterministic', type=cls.bool_type, default=False,
                help='Switching this on is beneficial to reproducing the '
                     'experiments, but may slow down the speed.'),
            cls.command_option(
                '--cudnn_allow_tf32', type=cls.bool_type, default=False,
                help='Whether to allow using TensorFloat-32 tensor cores for '
                     'computation. Switching this on can speed up matmul and '
                     'convolution operations, but may slightly harm the '
                     'numerical precision of floating-point operation.')
        ])

        options['Mixed-precision settings'].extend([
            cls.command_option(
                '--enable_amp', type=cls.bool_type, default=False,
                help='Whether to enable automatic mixed-precision (AMP) for '
                     'speeding up, but may affect the performance.')
        ])

        options['Resume/fine-tune settings'].extend([
            cls.command_option(
                '--resume_path', type=str, default=None,
                help='Path to the checkpoint to resume training, from which '
                     'the model weights and the optimizer states will be '
                     'both loaded.'),
            cls.command_option(
                '--weight_path', type=str, default=None,
                help='Path to the checkpoint to fine-tune the model, from '
                     'which only the model weights will be loaded.')
        ])

        options['Basic settings'].extend([
            cls.command_option(
                '--seed', type=cls.int_type, default=0,
                help='Seed for reproducibility. Set as negative to disable.'),
            cls.command_option(
                '--batch_size', type=cls.int_type, default=1,
                help='Batch size used for training on each replica.'),
            cls.command_option(
                '--val_batch_size', type=cls.int_type, default=0,
                help='Batch size used for validation on each replica. If not '
                     'provided, it will be set the same as `batch_size`.'),
            cls.command_option(
                '--total_iters', type=cls.int_type, default=0,
                help='Number of training iterations.'),
            cls.command_option(
                '--total_img', type=cls.int_type, default=0,
                help='Number of running samples before terminating the job. '
                     'This field only takes effect when `total_iters` is not '
                     'provided.'),
            cls.command_option(
                '--total_epochs', type=cls.int_type, default=0,
                help='Number of training epochs. This field only takes effect '
                     'when neither `total_iters` nor `total_img` is provided.')
        ])

        options['Training dataset settings'].extend([
            cls.command_option(
                '--train_dataset', type=str, default=None,
                help='Path to the training dataset.'),
            cls.command_option(
                '--train_data_file_format', type=str, default=None,
                help='Format of how the training data is stored on the disk. '
                     'If left empty, the format will be auto-detected.'),
            cls.command_option(
                '--train_anno_path', type=str, default=None,
                help='Path to the annotation file of training data.'),
            cls.command_option(
                '--train_anno_meta', type=str, default=None,
                help='Name of the annotation meta within the training data.'),
            cls.command_option(
                '--train_anno_format', type=str, default=None,
                help='Format of the annotation file of training data. If left '
                     'empty, the format will be auto-detected.'),
            cls.command_option(
                '--train_max_samples', type=cls.int_type, default=-1,
                help='Maximum number of samples used from training data. '
                     'Non-positive means to use all samples.'),
            cls.command_option(
                '--train_data_mirror', type=cls.bool_type, default=False,
                help='Whether to mirror the training dataset by flipping each '
                     'sample horizontally to enlarge the dataset twice.')
        ])

        options['Validation dataset settings'].extend([
            cls.command_option(
                '--val_dataset', type=str, default=None,
                help='Path to the validation dataset.'),
            cls.command_option(
                '--val_data_file_format', type=str, default=None,
                help='Format of how the validation dataset is stored on the '
                     'disk. If left empty, the format will be auto-detected.'),
            cls.command_option(
                '--val_anno_path', type=str, default=None,
                help='Path to the annotation file of validation data.'),
            cls.command_option(
                '--val_anno_meta', type=str, default=None,
                help='Name of the annotation meta within the validation data.'),
            cls.command_option(
                '--val_anno_format', type=str, default=None,
                help='Format of the annotation file of validation data. If '
                     'left empty, the format will be auto-detected.'),
            cls.command_option(
                '--val_max_samples', type=cls.int_type, default=-1,
                help='Maximum number of samples used from validation data. '
                     'Non-positive means to use all samples.'),
            cls.command_option(
                '--val_data_mirror', type=cls.bool_type, default=False,
                help='Whether to mirror the validation dataset by flipping '
                     'each sample horizontally to enlarge the dataset twice.')
        ])

        options['Data loader settings'].extend([
            cls.command_option(
                '--data_loader_type', type=str, default='iter',
                help='Type of data loader.'),
            cls.command_option(
                '--data_repeat', type=cls.int_type, default=1,
                help='Times to repeat the data item list to save I/O time. '
                     'A too large number can be memory consuming. This field '
                     'only takes effect on the training dataset.'),
            cls.command_option(
                '--data_workers', type=cls.int_type, default=4,
                help='Number of data workers on each replica.'),
            cls.command_option(
                '--data_prefetch_factor', type=cls.int_type, default=2,
                help='Number of samples loaded in advance by each worker.'),
            cls.command_option(
                '--data_pin_memory', type=cls.bool_type, default=True,
                help='Whether to use pinned memory for loaded data. Switching '
                     'this on can make it faster to move data from CPU to GPU, '
                     'but may require a high-performance computing system. '
                     'This field only takes effect when `data_loader_type` is '
                     'set as `iter`.'),
            cls.command_option(
                '--data_threads', type=cls.int_type, default=4,
                help='Number of threads on each replica for data preparation. '
                     'This field only takes effect when `data_loader_type` is '
                     'set as `dali`.')
        ])

        options['Controller settings'].extend([
            cls.command_option(
                '--log_interval', type=cls.int_type, default=100,
                help='Interval (in iterations) of printing log.'),
            cls.command_option(
                '--ckpt_interval', type=cls.int_type, default=10000,
                help='Interval (in iterations) of saving checkpoint.'),
            cls.command_option(
                '--keep_ckpt_num', type=cls.int_type, default=-1,
                help='How many most recent checkpoints to keep. Set as '
                     'non-positive to keep all checkpoints.'),
            cls.command_option(
                '--save_running_metadata', type=cls.bool_type, default=True,
                help='Whether to save the running metadata, such as '
                     'batch size, current iteration, etc.'),
            cls.command_option(
                '--save_optimizer', type=cls.bool_type, default=True,
                help='Whether to save the optimizer.'),
            cls.command_option(
                '--save_learning_rate', type=cls.bool_type, default=True,
                help='Whether to save the learning rate.'),
            cls.command_option(
                '--save_loss', type=cls.bool_type, default=True,
                help='Whether to save the loss.'),
            cls.command_option(
                '--save_augment', type=cls.bool_type, default=True,
                help='Whether to save the augmentation pipeline.'),
            cls.command_option(
                '--save_running_stats', type=cls.bool_type, default=False,
                help='Whether to save the running stats.'),
            cls.command_option(
                '--eval_interval', type=cls.int_type, default=10000,
                help='Default interval to execute evaluation, which can be '
                     'overwritten by each individual metric.'),
            cls.command_option(
                '--eval_at_start', type=cls.bool_type, default=True,
                help='Default setting on whether to execute evaluation after '
                     'the first iteration, which can be overwritten by each '
                     'individual metric. This can help record the initial '
                     'performance, which is useful for tracking the training '
                     'process, and also help ensure all metrics to behave well '
                     'during training. However, this may cost a lot of time '
                     'when debugging.'),
            cls.command_option(
                '--save_best_ckpt', type=cls.bool_type, default=True,
                help='Default setting on whether to save a checkpoint for the '
                     'best performance regarding each evaluation metric, which '
                     'can be overwritten by each individual metric.')
        ])

        options['Profiler settings'].extend([
            cls.command_option(
                '--enable_profiler', type=cls.bool_type, default=False,
                help='Whether to enable performance profiler. This is useful '
                     'for tracking the computing time cost by each operation, '
                     'but may slow down the running process when profiling.'),
            cls.command_option(
                '--profiler_schedule_wait', type=cls.int_type, default=1,
                help='Number of iterations before the profiler starts.'),
            cls.command_option(
                '--profiler_schedule_warmup', type=cls.int_type, default=1,
                help='Number of iterations to warmup the profiler.'),
            cls.command_option(
                '--profiler_schedule_active', type=cls.int_type, default=3,
                help='Number of iterations used for profiling.'),
            cls.command_option(
                '--profiler_schedule_repeat', type=cls.int_type, default=2,
                help='Number of profiling cycles.'),
        ])

        return options

    @classmethod
    def get_recommended_options(cls):
        """Gets options that are commonly used."""
        recommended_options = [
            'job_name', 'enable_amp', 'resume_path', 'weight_path', 'seed',
            'batch_size', 'val_batch_size', 'data_loader_type', 'data_repeat',
            'total_img', 'total_epochs', 'total_iters', 'train_dataset',
            'train_anno_path', 'train_anno_meta', 'train_max_samples',
            'train_data_mirror', 'val_dataset', 'val_anno_path',
            'val_anno_meta', 'val_max_samples', 'val_data_mirror',
            'log_interval', 'ckpt_interval', 'keep_ckpt_num', 'eval_interval',
            'eval_at_start'
        ]
        return recommended_options

    def __init__(self, kwargs):
        """Initializes the class.

        Args:
            kwargs: A dictionary representing the keyword arguments, which is
                obtained from the command line. Please refer to function
                `self.get_command()` for details.
        """
        assert isinstance(kwargs, dict)
        self.args = kwargs  # Command arguments.
        self.prefetch_set = set()  # Collection of data to prefetch.

        self.config = EasyDict()
        self.config.command_name = self.name
        self.config.config_type = self.__class__.__name__
        self.config.source_file = os.path.relpath(
            sys.modules[self.__class__.__module__].__file__)

        self.config.runner_type = None  # Should be overwritten.
        self.config.data = dict()  # Should be overwritten.
        self.config.aug = dict(aug_type='NoAug')  # Can be overwritten.
        self.config.object_aug = dict(aug_type='NoAug')  # Can be overwritten.
        self.config.aug_kwargs = dict()  # Can be overwritten.
        self.config.object_aug_kwargs = dict()  # Can be overwritten.
        self.config.models = dict()  # Should be overwritten.
        self.config.loss = dict()  # Should be overwritten.
        self.config.metrics = dict()  # Can be overwritten.
        self.config.controllers = dict()  # Can be overwritten.

    def add_prefetch_file(self, path):
        """Adds a file that requires prefetching.

        Args:
            path: Path to the file that required prefetching.

        Returns:
            A new path, with the same basename as the input, locating at the
                data directory under the working directory. Such a path
                conversion is necessary since runners will only fetch data from
                the data directory.
        """
        return path

        if not path:
            return None

        self.prefetch_set.add(path)
        filename = os.path.basename(path)
        return os.path.join(
            self.config.work_dir, self.config.data_dir, filename)

    def parse_options(self):
        """Parses configurations from command options."""

        self.config.root_work_dir = self.args.pop('root_work_dir')
        self.config.job_name = self.args.pop('job_name')
        self.config.data_dir = self.args.pop('data_dir')
        self.config.checkpoint_dir = self.args.pop('checkpoint_dir')
        self.config.result_dir = self.args.pop('result_dir')
        self.config.tensorboard_dir = self.args.pop('tensorboard_dir')
        self.config.profile_dir = self.args.pop('profile_dir')
        self.config.resource_dir = self.args.pop('resource_dir')
        self.config.config_filename = self.args.pop('config_path')
        self.config.log_data_filename = self.args.pop('log_data_path')
        self.config.log_filename = self.args.pop('log_path')
        self.config.work_dir = os.path.join(
            self.config.root_work_dir, self.config.job_name)
        self.config.interactive = self.args.pop('interactive')

        self.config.logger_type = self.args.pop('logger_type')
        self.config.use_tensorboard = self.args.pop('use_tensorboard')
        self.config.file_transmitter_type = self.args.pop(
            'file_transmitter_type')
        file_transmitter_kwargs = self.args.pop('file_transmitter_kwargs')
        if file_transmitter_kwargs is None:
            self.config.file_transmitter_kwargs = dict()
        else:
            assert isinstance(file_transmitter_kwargs, dict)
            self.config.file_transmitter_kwargs = file_transmitter_kwargs

        self.config.enable_cudnn = self.args.pop('enable_cudnn')
        self.config.cudnn_benchmark = self.args.pop('cudnn_benchmark')
        self.config.cudnn_deterministic = self.args.pop('cudnn_deterministic')
        self.config.cudnn_allow_tf32 = self.args.pop('cudnn_allow_tf32')

        self.config.enable_amp = self.args.pop('enable_amp')

        self.config.resume_path = self.add_prefetch_file(
            self.args.pop('resume_path'))
        self.config.weight_path = self.add_prefetch_file(
            self.args.pop('weight_path'))

        self.config.seed = self.args.pop('seed')
        self.config.batch_size = self.args.pop('batch_size')
        self.config.val_batch_size = self.args.pop('val_batch_size')
        if self.config.val_batch_size <= 0:
            self.config.val_batch_size = self.config.batch_size
        self.config.total_iters = self.args.pop('total_iters')
        self.config.total_epochs = self.args.pop('total_epochs')
        self.config.total_img = self.args.pop('total_img')

        train_dataset = self.add_prefetch_file(self.args.pop('train_dataset'))
        train_anno_path = self.add_prefetch_file(
            self.args.pop('train_anno_path'))
        val_dataset = self.add_prefetch_file(self.args.pop('val_dataset'))
        val_anno_path = self.add_prefetch_file(self.args.pop('val_anno_path'))
        self.config.data.update(
            train=dict(
                dataset_type=None,
                root_dir=train_dataset,
                file_format=self.args.pop('train_data_file_format'),
                annotation_path=train_anno_path,
                annotation_meta=self.args.pop('train_anno_meta'),
                annotation_format=self.args.pop('train_anno_format'),
                max_samples=self.args.pop('train_max_samples'),
                mirror=self.args.pop('train_data_mirror'),
                transform_kwargs=None
            ),
            val=dict(
                dataset_type=None,
                root_dir=val_dataset,
                file_format=self.args.pop('val_data_file_format'),
                annotation_path=val_anno_path,
                annotation_meta=self.args.pop('val_anno_meta'),
                annotation_format=self.args.pop('val_anno_format'),
                max_samples=self.args.pop('val_max_samples'),
                mirror=self.args.pop('val_data_mirror'),
                transform_kwargs=None
            ),
            loader=dict(
                data_loader_type=self.args.pop('data_loader_type'),
                repeat=self.args.pop('data_repeat'),
                seed=self.config.seed,
                num_workers=self.args.pop('data_workers'),
                prefetch_factor=self.args.pop('data_prefetch_factor'),
                pin_memory=self.args.pop('data_pin_memory'),
                num_threads=self.args.pop('data_threads')
            )
        )

        log_interval = self.args.pop('log_interval')
        if log_interval > 0:
            self.config.controllers.update(
                RunningLogger=dict(every_n_iters=log_interval)
            )

        ckpt_interval = self.args.pop('ckpt_interval')
        save_running_metadata = self.args.pop('save_running_metadata')
        save_optimizer = self.args.pop('save_optimizer')
        save_learning_rate = self.args.pop('save_learning_rate')
        save_loss = self.args.pop('save_loss')
        save_augment = self.args.pop('save_augment')
        save_running_stats = self.args.pop('save_running_stats')

        if ckpt_interval > 0:
            self.config.controllers.update(
                Checkpointer=dict(
                    every_n_iters=ckpt_interval,
                    first_iter=True,
                    keep_ckpt_num=self.args.pop('keep_ckpt_num'),
                    save_running_metadata=save_running_metadata,
                    save_optimizer=save_optimizer,
                    save_learning_rate=save_learning_rate,
                    save_loss=save_loss,
                    save_augment=save_augment,
                    save_running_stats=save_running_stats
                )
            )

        self.config.controllers.update(
            Evaluator=dict(
                default_eval_interval=self.args.pop('eval_interval'),
                default_eval_at_start=self.args.pop('eval_at_start'),
                default_save_best_ckpt=self.args.pop('save_best_ckpt'),
                default_save_running_metadata=save_running_metadata,
                default_save_optimizer=save_optimizer,
                default_save_learning_rate=save_learning_rate,
                default_save_loss=save_loss,
                default_save_augment=save_augment,
                default_save_running_stats=save_running_stats
            )
        )

        self.config.enable_profiler = self.args.pop('enable_profiler')
        self.config.profiler_schedule_kwargs = dict(
            wait=self.args.pop('profiler_schedule_wait'),
            warmup=self.args.pop('profiler_schedule_warmup'),
            active=self.args.pop('profiler_schedule_active'),
            repeat=self.args.pop('profiler_schedule_repeat')
        )

    def update_config(self):
        """Updates the configuration with key-value pairs.

        Such a update is in a hierarchical manner. For example, for key-value
        `a.b.c=v` in `self.args.options`, `self.config` will be
        updated by `self.config['a']['b']['c'] = v`.
        """
        options = self.args.pop('options', list())
        for option in options:
            key, val = option.split('=', maxsplit=1)
            hierarchical_keys = key.split('.')
            temp = self.config
            for sub_key in hierarchical_keys[:-1]:
                temp = temp[sub_key]
            temp[hierarchical_keys[-1]] = parse_json(val)

    def get_config(self):
        """Gets the complete configurations.

        Basically, this function wraps `self.parse_options()` and
        `self.update_config()`.
        """
        # Parse configuration.
        self.parse_options()
        self.update_config()
        self.config.prefetch_list = list(self.prefetch_set)

        assert self.config.runner_type, 'Runner type is missing!'
        assert self.config.data.train.dataset_type, 'Dataset type is missing!'
        assert self.config.data.val.dataset_type, 'Dataset type is missing!'
        assert len(self.args) == 0, f'Fail to parse `{self.args}`!'
        return self.config
