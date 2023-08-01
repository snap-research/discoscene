# python3.7
"""Contains the class for profiling."""

import torch
from torch import distributed as dist

from utils.tf_utils import import_tb_writer

SummaryWriter = import_tb_writer()

__all__ = ['Profiler']


class Profiler(object):
    """Defines the profiler.

    Essentially, this is a wrapper of `torch.profiler.profile`.
    If `enable` is set to `False`, this profiler becomes a dummy context manager
    with a dummy `step()`.

    Args:
        enable: `bool`, whether to enable `torch.profiler`.
        tb_dir: `str`, path to save profiler's TensorBoard events file.
        logger: `utils.loggers` or `logging.Logger`, the event logging system.
        **schedule_kwargs: settings to the `schedule` of
            `torch.profiler.profile`. The profiler will skip the first
            `skip_first` steps, then wait for `wait` steps, then do the warmup
            for the next `warmup` steps, then do the active recording for the
            next `active` steps and then repeat the cycle starting with `wait`
            steps. (default: dict(wait=1, warmup=1, active=3, repeat=2))
    """
    def __init__(self,
                 enable=False,
                 tb_dir='.',
                 logger=None,
                 **schedule_kwargs):
        self.enable = enable
        rank = dist.get_rank() if dist.is_initialized() else 0
        if enable and SummaryWriter is not None:
            try:  # In case the PyTorch version is outdated.
                if rank == 0:
                    SummaryWriter(tb_dir)
                if dist.is_initialized():
                    dist.barrier()  # Only create one TensorBoard event.
                if schedule_kwargs is None:
                    schedule_kwargs = dict(wait=1, warmup=1, active=3, repeat=2)
                # Profile CPU and each GPU.
                self.profiler = torch.profiler.profile(
                    schedule=torch.profiler.schedule(**schedule_kwargs),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        tb_dir),
                    record_shapes=True,
                    with_stack=True)
                if logger:
                    logger.info(f'Enable profiler with schedule: '
                                f'{schedule_kwargs}.\n')
            except AttributeError as error:
                logger.warning(f'Skipping profiler due to {error}!\n'
                               f'Please update your PyTorch to 1.8.1 or '
                               f'later to enable profiler.\n')
                self.enable = False

    def __enter__(self):
        if self.enable:
            return self.profiler.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enable:
            self.profiler.__exit__(exc_type, exc_val, exc_tb)

    def step(self):
        """Executes the profiler for one step."""
        if self.enable:
            self.profiler.step()
