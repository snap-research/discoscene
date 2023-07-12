# python3.7
"""Collects all loggers."""

from .normal_logger import NormalLogger
from .rich_logger import RichLogger
from .dummy_logger import DummyLogger

__all__ = ['build_logger']

_LOGGERS = {
    'normal': NormalLogger,
    'rich': RichLogger,
    'dummy': DummyLogger
}


def build_logger(logger_type='normal', **kwargs):
    """Builds a logger.

    Args:
        logger_type: Type of logger, which is case insensitive.
            (default: `normal`)
        **kwargs: Additional arguments to build the logger.

    Raises:
        ValueError: If the `logger_type` is not supported.
    """
    logger_type = logger_type.lower()
    if logger_type not in _LOGGERS:
        raise ValueError(f'Invalid logger type: `{logger_type}`!\n'
                         f'Types allowed: {list(_LOGGERS)}.')
    return _LOGGERS[logger_type](**kwargs)
