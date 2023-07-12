# python3.7
"""Contains the running controller to record time."""

import time

from .base_controller import BaseController

__all__ = ['Timer']


class Timer(BaseController):
    """Defines the running controller to record running time.

    This controller will be executed every iteration (both before and after) to
    summarize the data preparation time as well as the model running time.
    Besides, this controller will also mark the start and end time of the
    running process.

    NOTE: This controller is set to `LOW` priority by default and will only be
    executed on the chief worker.
    """

    def __init__(self, config=None):
        config = config or dict()
        config.setdefault('priority', 'LOW')
        config.setdefault('every_n_iters', 1)
        config.setdefault('chief_only', True)
        super().__init__(config)

        self.time = time.time()

    def setup(self, runner):
        runner.running_stats.add(
            'data time', log_format='time', requires_sync=False)
        runner.running_stats.add(
            'iter time', log_format='time', requires_sync=False)
        runner.running_stats.add('run time',
                                 log_format='time',
                                 log_strategy='CURRENT',
                                 requires_sync=False)
        self.time = time.time()
        runner.start_time = self.time

    def close(self, runner):
        runner.end_time = time.time()

    def execute_before_iteration(self, runner):
        start_time = time.time()
        runner.running_stats.update({'data time': start_time - self.time})

    def execute_after_iteration(self, runner):
        end_time = time.time()
        runner.running_stats.update({'iter time': end_time - self.time})
        runner.running_stats.update({'run time': end_time - runner.start_time})
        self.time = end_time
