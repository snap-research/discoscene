# python3.7
"""Contains the class of normal logger.

This class is built based on the built-in function `print()`, the module
`logging` and the module `tqdm` for progressive bar.
"""

import sys
import logging
from copy import deepcopy
from tqdm import tqdm

from .base_logger import BaseLogger

__all__ = ['NormalLogger']


class NormalLogger(BaseLogger):
    """Implements the logger based on `logging` module."""

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
        formatter = logging.Formatter(
            '[%(asctime)s][%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')

        # Print log message onto the screen.
        terminal_handler = logging.StreamHandler(stream=sys.stdout)
        terminal_handler.setLevel(self.screen_level)
        terminal_handler.setFormatter(formatter)
        self.logger.addHandler(terminal_handler)

        # Save log message into log file if needed.
        if self.logfile:
            # File will be closed when the logger is closed in `self.close()`.
            self.file_stream = open(self.logfile, 'a')  # pylint: disable=consider-using-with
            file_handler = logging.StreamHandler(stream=self.file_stream)
            file_handler.setLevel(self.file_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        self.pbar = []
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
            print(*messages, file=handler.stream)

    def init_pbar(self, leave=False):
        columns = [
            '{desc}',
            '{bar}',
            ' {percentage:5.1f}%',
            '[{elapsed}<{remaining}, {rate_fmt}{postfix}]',
        ]
        self.pbar_kwargs = dict(
            leave=leave,
            bar_format=' '.join(columns),
            unit='',
        )

    def add_pbar_task(self, name, total, **kwargs):
        assert isinstance(self.pbar_kwargs, dict)
        pbar_kwargs = deepcopy(self.pbar_kwargs)
        pbar_kwargs.update(**kwargs)
        self.pbar.append(tqdm(desc=name, total=total, **pbar_kwargs))
        return len(self.pbar) - 1

    def update_pbar(self, task_id, advance=1):
        assert len(self.pbar) > task_id and isinstance(self.pbar[task_id], tqdm)
        if self.pbar[task_id].n < self.pbar[task_id].total:
            self.pbar[task_id].update(advance)
            if self.pbar[task_id].n >= self.pbar[task_id].total:
                self.pbar[task_id].refresh()

    def close_pbar(self):
        for pbar in self.pbar[::-1]:
            pbar.close()
        self.pbar = []
        self.pbar_kwargs = {}
