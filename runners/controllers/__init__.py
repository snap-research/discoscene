# python3.7
"""Collects all controllers."""

from .ada_aug_controller import AdaAugController
from .object_ada_aug_controller import ObjectAdaAugController
from .batch_visualizer import BatchVisualizer
from .cache_cleaner import CacheCleaner
from .checkpointer import Checkpointer
from .dataset_visualizer import DatasetVisualizer
from .evaluator import Evaluator
from .lr_scheduler import LRScheduler
from .progress_scheduler import ProgressScheduler
from .progress_schedulerv2 import ProgressSchedulerV2
from .running_logger import RunningLogger
from .timer import Timer

__all__ = ['build_controller']

_CONTROLLERS = {
    'AdaAugController': AdaAugController,
    'ObjectAdaAugController': ObjectAdaAugController,
    'BatchVisualizer': BatchVisualizer,
    'CacheCleaner': CacheCleaner,
    'Checkpointer': Checkpointer,
    'DatasetVisualizer': DatasetVisualizer,
    'Evaluator': Evaluator,
    'LRScheduler': LRScheduler,
    'ProgressScheduler': ProgressScheduler,
    'RunningLogger': RunningLogger,
    'Timer': Timer,
    'ProgressSchedulerV2': ProgressSchedulerV2,
}


def build_controller(controller_type, config=None):
    """Builds a controller based on its class type.

    Args:
        controller_type: Class type to which the controller belongs, which is
            case sensitive.
        config: Configuration of the controller. (default: None)

    Raises:
        ValueError: If the `controller_type` is not supported.
    """
    if controller_type not in _CONTROLLERS:
        raise ValueError(f'Invalid controller type: `{controller_type}`!\n'
                         f'Types allowed: {list(_CONTROLLERS)}.')
    return _CONTROLLERS[controller_type](config)
