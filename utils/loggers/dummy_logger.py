# python3.7
"""Contains the class of dummy logger.

This logger has all expected logging functions but behaves silently, which is
very useful in multi-processing mode. Only the chief process can have the logger
with normal behavior.
"""

from .base_logger import BaseLogger

__all__ = ['DummyLogger']


class DummyLogger(BaseLogger):
    """Implements a dummy logger which logs nothing."""

    def __init__(self,
                 logger_name='logger',
                 logfile=None,
                 screen_level=None,
                 file_level=None,
                 indent_space=4,
                 verbose_log=False):
        super().__init__(logger_name=logger_name,
                         logfile=logfile,
                         screen_level=screen_level,
                         file_level=file_level,
                         indent_space=indent_space,
                         verbose_log=verbose_log)

    def _log(self, message, **kwargs):
        return

    def _debug(self, message, **kwargs):
        return

    def _info(self, message, **kwargs):
        return

    def _warning(self, message, **kwargs):
        return

    def _error(self, message, **kwargs):
        return

    def _exception(self, message, **kwargs):
        return

    def _critical(self, message, **kwargs):
        return

    def _print(self, *messages, **kwargs):
        return

    def init_pbar(self, leave=False):
        return

    def add_pbar_task(self, name, total, **kwargs):
        return -1

    def update_pbar(self, task_id, advance=1):
        return

    def close_pbar(self):
        return
