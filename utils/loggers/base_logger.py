# python3.7
"""Contains the base class for logging.

Basically, this is an interface bridging the program and the local file system.
A logger is able to log wrapped message onto the screen and a log file.
"""

import logging

__all__ = ['BaseLogger']


class BaseLogger(object):
    """Defines the base logger.

    A logger should have the following members:

    (1) logger: The logger to record message.
    (2) pbar: The progressive bar (shown on the screen only).
    (3) pbar_kwargs: The arguments for the progressive bar.
    (4) file_stream: The stream to log messages into if needed.

    A logger should have the following functions:

    (1) log(): The base function to log message.
    (2) debug(): The function to log message with `DEBUG` level.
    (3) info(): The function to log message with `INFO` level.
    (4) warning(): The function to log message with `WARNING` level.
    (5) warn(): Same as function `warning()`.
    (6) error(): The function to log message with `ERROR` level.
    (7) exception(): The function to log message with exception information.
    (8) critical(): The function to log message with `CRITICAL` level.
    (9) fatal(): Same as function `critical()`.
    (10) print(): The function to print the message without any decoration.
    (11) init_pbar(): The function to initialize the progressive bar.
    (12) add_pbar_task(): The function to add a task to the progressive bar.
    (13) update_pbar(): The function to update the progressive bar.
    (14) close_pbar(): The function to close the progressive bar.

    The logger will record log message both on screen and to file.

    Args:
        logger_name: Unique name for the logger. (default: `logger`)
        logfile: Path to the log file. If set as `None`, the file stream
            will be skipped. (default: `None`)
        screen_level: Minimum level of message to log onto screen.
            (default: `logging.INFO`)
        file_level: Minimum level of message to log into file.
            (default: `logging.DEBUG`)
        indent_space: Number of spaces between two adjacent indent levels.
            (default: 4)
        verbose_log: Whether to log verbose message. (default: False)
    """

    def __init__(self,
                 logger_name='logger',
                 logfile=None,
                 screen_level=logging.INFO,
                 file_level=logging.DEBUG,
                 indent_space=4,
                 verbose_log=False):
        self.logger_name = logger_name
        self.logfile = logfile
        self.screen_level = screen_level
        self.file_level = file_level
        self.indent_space = indent_space
        self.verbose_log = verbose_log

        self.logger = None
        self.pbar = None
        self.pbar_kwargs = None
        self.file_stream = None

        self.warn = self.warning
        self.fatal = self.critical

    def __del__(self):
        self.close()

    def close(self):
        """Closes the logger."""
        if self.file_stream is not None:
            self.file_stream.close()

    @property
    def name(self):
        """Returns the class name of the logger."""
        return self.__class__.__name__

    # Log message.
    def wrap_message(self, message, indent_level=0):
        """Wraps the message with indent."""
        if message is None:
            message = ''
        assert isinstance(message, str)
        assert isinstance(indent_level, int) and indent_level >= 0
        if message == '':
            return ''
        return ' ' * (indent_level * self.indent_space) + message

    def _log(self, message, **kwargs):
        """Logs wrapped message."""
        raise NotImplementedError('Should be implemented in derived class!')

    def _debug(self, message, **kwargs):
        """Logs wrapped message with `DEBUG` level."""
        raise NotImplementedError('Should be implemented in derived class!')

    def _info(self, message, **kwargs):
        """Logs wrapped message with `INFO` level."""
        raise NotImplementedError('Should be implemented in derived class!')

    def _warning(self, message, **kwargs):
        """Logs wrapped message with `WARNING` level."""
        raise NotImplementedError('Should be implemented in derived class!')

    def _error(self, message, **kwargs):
        """Logs wrapped message with `ERROR` level."""
        raise NotImplementedError('Should be implemented in derived class!')

    def _exception(self, message, **kwargs):
        """Logs wrapped message with exception information."""
        raise NotImplementedError('Should be implemented in derived class!')

    def _critical(self, message, **kwargs):
        """Logs wrapped message with `CRITICAL` level."""
        raise NotImplementedError('Should be implemented in derived class!')

    def _print(self, *messages, **kwargs):
        """Prints wrapped message without any decoration."""
        raise NotImplementedError('Should be implemented in derived class!')

    def log(self, message, indent_level=0, is_verbose=False, **kwargs):
        """Logs message.

        The message is wrapped with indent, and will be disabled if `is_verbose`
        is set as `True`.
        """
        if is_verbose and not self.verbose_log:
            return
        message = self.wrap_message(message, indent_level=indent_level)
        self._log(message, **kwargs)

    def debug(self, message, indent_level=0, is_verbose=False, **kwargs):
        """Logs message with `DEBUG` level.

        The message is wrapped with indent, and will be disabled if `is_verbose`
        is set as `True`.
        """
        if is_verbose and not self.verbose_log:
            return
        message = self.wrap_message(message, indent_level=indent_level)
        self._debug(message, **kwargs)

    def info(self, message, indent_level=0, is_verbose=False, **kwargs):
        """Logs message with `INFO` level.

        The message is wrapped with indent, and will be disabled if `is_verbose`
        is set as `True`.
        """
        if is_verbose and not self.verbose_log:
            return
        message = self.wrap_message(message, indent_level=indent_level)
        self._info(message, **kwargs)

    def warning(self, message, indent_level=0, is_verbose=False, **kwargs):
        """Logs message with `WARNING` level.

        The message is wrapped with indent, and will be disabled if `is_verbose`
        is set as `True`.
        """
        if is_verbose and not self.verbose_log:
            return
        message = self.wrap_message(message, indent_level=indent_level)
        self._warning(message, **kwargs)

    def error(self, message, indent_level=0, is_verbose=False, **kwargs):
        """Logs message with `ERROR` level.

        The message is wrapped with indent, and will be disabled if `is_verbose`
        is set as `True`.
        """
        if is_verbose and not self.verbose_log:
            return
        message = self.wrap_message(message, indent_level=indent_level)
        self._error(message, **kwargs)

    def exception(self, message, indent_level=0, is_verbose=False, **kwargs):
        """Logs message with exception information.

        The message is wrapped with indent, and will be disabled if `is_verbose`
        is set as `True`.
        """
        if is_verbose and not self.verbose_log:
            return
        message = self.wrap_message(message, indent_level=indent_level)
        self._exception(message, **kwargs)

    def critical(self, message, indent_level=0, is_verbose=False, **kwargs):
        """Logs message with `CRITICAL` level.

        The message is wrapped with indent, and will be disabled if `is_verbose`
        is set as `True`.
        """
        if is_verbose and not self.verbose_log:
            return
        message = self.wrap_message(message, indent_level=indent_level)
        self._critical(message, **kwargs)

    def print(self, *messages, indent_level=0, is_verbose=False, **kwargs):
        """Prints message without any decoration.

        The message is wrapped with indent, and will be disabled if `is_verbose`
        is set as `True`.
        """
        if is_verbose and not self.verbose_log:
            return
        new_messages = []
        for message in messages:
            new_messages.append(
                self.wrap_message(message, indent_level=indent_level))
        self._print(*new_messages, **kwargs)

    # Progressive bar.
    def init_pbar(self, leave=False):
        """Initializes the progressive bar.

        Args:
            leave: Whether to leave the trace of the progressive bar.
                (default: False)
        """
        raise NotImplementedError('Should be implemented in derived class!')

    def add_pbar_task(self, name, total, **kwargs):
        """Adds a task to the progressive bar.

        Args:
            name: Name of the added task.
            total: Total number of steps (samples) contained in the task.
            **kwargs: Additional arguments.

        Returns:
            Task ID.
        """
        raise NotImplementedError('Should be implemented in derived class!')

    def update_pbar(self, task_id, advance=1):
        """Updates the progressive bar.

        Args:
            task_id: ID of the task to update.
            advance: Number of steps advanced onto the target task. (default: 1)
        """
        raise NotImplementedError('Should be implemented in derived class!')

    def close_pbar(self):
        """Closes the progress bar."""
        raise NotImplementedError('Should be implemented in derived class!')
