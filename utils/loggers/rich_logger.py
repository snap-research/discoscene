# python3.7
"""Contains the class of rich logger.

This class is based on the module `rich`. Please refer to
https://github.com/Textualize/rich for more details.
"""

import sys
import logging
from copy import deepcopy
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress
from rich.progress import ProgressColumn
from rich.progress import TextColumn
from rich.progress import BarColumn
from rich.text import Text

from .base_logger import BaseLogger

__all__ = ['RichLogger']


def _format_time(seconds):
    """Formats seconds to readable time string.

    This function is used to display time in progress bar.
    """
    if not seconds:
        return '--:--'

    seconds = int(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f'{hours}:{minutes:02d}:{seconds:02d}'
    return f'{minutes:02d}:{seconds:02d}'


class TimeColumn(ProgressColumn):
    """Renders total time, ETA, and speed in progress bar."""

    max_refresh = 0.5  # Only refresh twice a second to prevent jitter

    def render(self, task):
        elapsed_time = _format_time(task.elapsed)
        eta = _format_time(task.time_remaining)
        speed = f'{task.speed:.2f}/s' if task.speed else '?/s'
        return Text(f'[{elapsed_time}<{eta}, {speed}]',
                    style='progress.remaining')


class RichLogger(BaseLogger):
    """Implements the logger based on `rich` module."""

    def __init__(self,
                 logger_name='logger',
                 logfile=None,
                 screen_level=logging.INFO,
                 file_level=logging.DEBUG,
                 indent_space=4,
                 verbose_log=False):
        super().__init__(logger_name=logger_name,
                         logfile=logfile,
                         screen_level=screen_level,
                         file_level=file_level,
                         indent_space=indent_space,
                         verbose_log=verbose_log)

        # Get logger and check whether the logger has already been created.
        self.logger = logging.getLogger(self.logger_name)
        self.logger.propagate = False
        if self.logger.hasHandlers():  # Already existed
            raise SystemExit(f'Logger `{self.logger_name}` has already '
                             f'existed!\n'
                             f'Please use another name, or otherwise the '
                             f'messages may be mixed up.')

        # Set format.
        self.logger.setLevel(logging.DEBUG)

        # Print log message onto the screen.
        terminal_console = Console(
            file=sys.stdout, log_time=False, log_path=False)
        terminal_handler = RichHandler(
            level=self.screen_level,
            console=terminal_console,
            show_time=True,
            show_level=True,
            show_path=False,
            log_time_format='[%Y-%m-%d %H:%M:%S] ')
        terminal_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(terminal_handler)

        # Save log message into log file if needed.
        if self.logfile:
            # File will be closed when the logger is closed in `self.close()`.
            self.file_stream = open(self.logfile, 'a')  # pylint: disable=consider-using-with
            file_console = Console(
                file=self.file_stream, log_time=False, log_path=False)
            file_handler = RichHandler(
                level=self.file_level,
                console=file_console,
                show_time=True,
                show_level=True,
                show_path=False,
                log_time_format='[%Y-%m-%d %H:%M:%S] ')
            file_handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(file_handler)

        self.pbar = None
        self.pbar_kwargs = {}

    def _log(self, message, **kwargs):
        self.logger.log(message, **kwargs)

    def _debug(self, message, **kwargs):
        self.logger.debug(message, **kwargs)

    def _info(self, message, **kwargs):
        self.logger.info(message, **kwargs)

    def _warning(self, message, **kwargs):
        self.logger.warning(message, **kwargs)

    def _error(self, message, **kwargs):
        self.logger.error(message, **kwargs)

    def _exception(self, message, **kwargs):
        self.logger.exception(message, **kwargs)

    def _critical(self, message, **kwargs):
        self.logger.critical(message, **kwargs)

    def _print(self, *messages, **kwargs):
        for handler in self.logger.handlers:
            handler.console.print(*messages, **kwargs)

    def init_pbar(self, leave=False):
        assert self.pbar is None

        # Columns shown in the progress bar.
        columns = (
            TextColumn('[progress.description]{task.description}'),
            BarColumn(bar_width=None),
            TextColumn('[progress.percentage]{task.percentage:>5.1f}%'),
            TimeColumn(),
        )

        self.pbar = Progress(*columns,
                             console=self.logger.handlers[0].console,
                             transient=not leave,
                             auto_refresh=True,
                             refresh_per_second=10)
        self.pbar.start()

    def add_pbar_task(self, name, total, **kwargs):
        assert isinstance(self.pbar, Progress)
        assert isinstance(self.pbar_kwargs, dict)
        pbar_kwargs = deepcopy(self.pbar_kwargs)
        pbar_kwargs.update(**kwargs)
        task_id = self.pbar.add_task(name, total=total, **pbar_kwargs)
        return task_id

    def update_pbar(self, task_id, advance=1):
        assert isinstance(self.pbar, Progress)
        if self.pbar.tasks[task_id].finished:
            if self.pbar.tasks[task_id].stop_time is None:
                self.pbar.stop_task(task_id)
        else:
            self.pbar.update(task_id, advance=advance)

    def close_pbar(self):
        assert isinstance(self.pbar, Progress)
        self.pbar.stop()
        self.pbar = None
        self.pbar_kwargs = {}
