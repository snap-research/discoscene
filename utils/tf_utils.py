# python3.7
"""Contains the utility functions to handle import TensorFlow modules.

Basically, TensorFlow may not be supported in the current environment, or may
cause some warnings. This file provides functions to help ease TensorFlow
related imports, such as TensorBoard.
"""

import warnings

__all__ = ['import_tf', 'import_tb_writer']


def import_tf():
    """Imports TensorFlow module if possible.

    If `ImportError` is raised, `None` will be returned. Otherwise, the module
    `tensorflow` will be returned.
    """
    warnings.filterwarnings('ignore', category=FutureWarning)
    try:
        import tensorflow as tf  # pylint: disable=import-outside-toplevel
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        module = tf
    except ImportError:
        module = None
    warnings.filterwarnings('default', category=FutureWarning)
    return module


def import_tb_writer():
    """Imports the SummaryWriter of TensorBoard.

    If `ImportError` is raised, `None` will be returned. Otherwise, the class
    `SummaryWriter` will be returned.

    NOTE: This function attempts to import `SummaryWriter` from
    `torch.utils.tensorboard`. But it does not necessarily mean the import
    always succeeds because installing TensorBoard is not a duty of `PyTorch`.
    """
    warnings.filterwarnings('ignore', category=FutureWarning)
    try:
        from torch.utils.tensorboard import SummaryWriter  # pylint: disable=import-outside-toplevel
    except ImportError:  # In case TensorBoard is not supported.
        SummaryWriter = None
    warnings.filterwarnings('default', category=FutureWarning)
    return SummaryWriter
