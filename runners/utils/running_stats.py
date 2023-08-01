# python3.7
"""Contains the class for recording the running stats.

Here, running stats refers to the statistical information in the running
process, such as loss values, learning rates, running time, etc.

NOTE: ONLY scalars are supported.
"""

import torch
import torch.distributed as dist

from utils.formatting_utils import format_time

__all__ = ['SingleStats', 'RunningStats']


class SingleStats(object):
    """A class to record the stats corresponding to a particular variable.

    This class is log-friendly and supports customized log format, including:

    (1) Numerical log format, such as `.3f`, `.1e`, `05d`, and `>10s`.
    (2) Customized log name (name of the stats to show in the log).
    (3) Additional string (e.g., measure unit) as the tail of log message.

    Furthermore, this class also supports logging the stats with different
    strategies, including:

    (1) CURRENT: The current value will be logged.
    (2) AVERAGE: The averaged value (from the beginning) will be logged.
    (3) CUMULATIVE: The cumulative value (from the beginning) will be logged.

    After instantiation, please use `self.update(value)` to update the stats
    with new data (e.g., iterative). Then, please use `self.summarize()` to get
    the long-term summary of the stats. This function will also do
    synchronization across different replicas if needed.
    """

    def __init__(self,
                 name,
                 log_format='.3f',
                 log_name=None,
                 log_tail=None,
                 log_strategy='AVERAGE',
                 requires_sync=True,
                 keep_previous=False):
        """Initializes the stats with log format.

        Args:
            name: Name of the stats. Should be a string without spaces.
            log_format: The numerical log format. Use `time` to log time
                duration. Use `None` to disable logging to screen and log file.
                (default: `.3f`)
            log_name: The name shown in the log. `None` means to directly use
                the stats name. (default: None)
            log_tail: The tailing log message. (default: None)
            log_strategy: Strategy to log this stats. `CURRENT`, `AVERAGE`, and
                `CUMULATIVE` are supported. (default: `AVERAGE`)
            requires_sync: Whether to synchronize across replicas. If not, the
                results from the chief process will be used. (default: True)
            keep_previous: Whether to keep the previous stats if no new data
                comes in within a period of time. If set as `True`, stats from
                previous timestamp will be used for logging. Otherwise, `0.0`
                will be logged. (default: False)

        Raises:
            ValueError: If the input `log_strategy` is not supported.
        """
        log_strategy = log_strategy.upper()
        if log_strategy not in ['CURRENT', 'AVERAGE', 'CUMULATIVE']:
            raise ValueError(f'Invalid log strategy `{self.log_strategy}`!')

        self._name = name
        self._log_format = log_format
        self._log_name = log_name or name
        self._log_tail = log_tail or ''
        self._log_strategy = log_strategy
        self._requires_sync = requires_sync
        self._keep_previous = keep_previous

        # Settings for distributed stats.
        self.is_distributed = dist.is_initialized()
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Init stats data.
        self.val = torch.zeros([], dtype=torch.float64)  # Current value.
        self.cum = torch.zeros([], dtype=torch.float64)  # Cumulative value.
        self.cnt = torch.zeros([], dtype=torch.float64)  # Count number.
        # Previous data.
        self.prev_val = torch.zeros([], dtype=torch.float64)
        self.prev_cum = torch.zeros([], dtype=torch.float64)
        self.prev_cnt = torch.zeros([], dtype=torch.float64)
        # Summarized data, which can be used for logging.
        self.summarized_val = None

    @property
    def name(self):
        """Gets the name of the stats."""
        return self._name

    @property
    def log_format(self):
        """Gets tne numerical log format of the stats."""
        return self._log_format

    @property
    def log_name(self):
        """Gets the log name of the stats."""
        return self._log_name

    @property
    def log_tail(self):
        """Gets the tailing log message of the stats."""
        return self._log_tail

    @property
    def log_strategy(self):
        """Gets the log strategy of the stats."""
        return self._log_strategy

    @property
    def requires_sync(self):
        """Gets the synchronize state of the stats."""
        return self._requires_sync

    @property
    def keep_previous(self):
        """Whether to use previous stats for logging."""
        return self._keep_previous

    def update(self, value):
        """Updates the stats data."""
        value = torch.as_tensor(value).cpu().detach().to(torch.float64)
        self.val = value.mean()
        self.cum = self.cum + value.sum()
        self.cnt = self.cnt + value.numel()

    def summarize(self):
        """Gets value for logging according to the log strategy."""
        if self.cnt > 0:  # Has new data coming in, using new data.
            val = self.val
            cum = self.cum
            cnt = self.cnt
            if self.keep_previous:  # Record history stats.
                self.prev_val = self.val
                self.prev_cum = self.cum
                self.prev_cnt = self.cnt
        else:  # No new data coming in, use history.
            val = self.prev_val
            cum = self.prev_cum
            cnt = self.prev_cnt

        if self.requires_sync and self.is_distributed:
            # NOTE: `torch.distributed.all_reduce()` may only work for GPU data.
            # Hence we move the data onto GPU for reducing.
            sync_tensor = torch.stack([val, cum, cnt]).cuda()
            dist.all_reduce(sync_tensor, op=dist.ReduceOp.SUM)
            sync_tensor = sync_tensor / self.world_size
            val, cum, cnt = sync_tensor.cpu()

        if self.log_strategy == 'CURRENT':
            self.summarized_val = float(val)
        elif self.log_strategy == 'AVERAGE':
            self.summarized_val = float(cum / cnt) if cnt > 0 else float(0.0)
        elif self.log_strategy == 'CUMULATIVE':
            self.summarized_val = float(cum)
        else:
            raise NotImplementedError(f'Log strategy `{self.log_strategy}` is '
                                      f'not implemented!')

        # Clear stats to only record stats for a period of time.
        self.val = torch.zeros([], dtype=torch.float64)
        self.cum = torch.zeros([], dtype=torch.float64)
        self.cnt = torch.zeros([], dtype=torch.float64)

        return self.summarized_val

    def __str__(self):
        """Gets log message."""
        val = self.summarized_val

        if self.log_format is None:
            return ''
        if self.log_format == 'time':
            val_str = f'{format_time(val)}'
        else:
            if 'd' in self.log_format:
                val = int(val + 0.5)
            val_str = f'{val:{self.log_format}}'
        return f'{self.log_name}: {val_str}{self.log_tail}'


class RunningStats(object):
    """A class to record all the running stats.

    Basically, this class contains a dictionary of SingleStats.

    Example:

    running_stats = RunningStats()
    running_stats.add('loss', log_format='.3f', log_strategy='AVERAGE')
    running_stats.add('time', log_format='time', log_name='Iter Time',
                      log_strategy='CURRENT')
    running_stats.log_order = ['time', 'loss']
    running_stats.update({'loss': 0.46, 'time': 12})
    running_stats.update({'time': 14.5, 'loss': 0.33})
    running_stats.summarize()
    print(running_stats)
    """

    def __init__(self, log_delimiter=', '):
        """Initializes the running stats with the log delimiter.

        Args:
            log_delimiter: This delimiter is used to connect the log messages
                from different stats. (default: `, `)
        """
        self._log_delimiter = log_delimiter
        self.stats_pool = dict()  # The stats pool.
        self.log_order = None  # Order of the stats to log.
        self.is_resumed = False  # Whether resumed from checkpoint.

    @property
    def log_delimiter(self):
        """Gets the log delimiter between different stats."""
        return self._log_delimiter

    def add(self, name, **kwargs):
        """Adds a new `SingleStats` to the dictionary."""
        if name in self.stats_pool:
            if self.is_resumed:  # skip if resumed
                return
            raise ValueError(f'Stats `{name}` has already existed!')
        self.stats_pool[name] = SingleStats(name, **kwargs)

    def update(self, kwargs):
        """Updates the stats data by name."""
        for name, value in kwargs.items():
            self.stats_pool[name].update(value)

    def summarize(self):
        """Summarizes all stats in the stats pool."""
        for stats in self.stats_pool.values():
            _ = stats.summarize()

    def __getattr__(self, name):
        """Gets a particular SingleStats by name."""
        if name in self.stats_pool:
            return self.stats_pool[name]
        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError(f'`{self.__class__.__name__}` object has no '
                             f'attribute `{name}`!')

    def __str__(self):
        """Gets log message."""
        log_order = self.log_order or list(self.stats_pool)
        log_strings = []
        for name in log_order:
            stats = self.stats_pool[name]
            if stats.log_format is not None:
                log_strings.append(str(stats))
        return self.log_delimiter.join(log_strings)
