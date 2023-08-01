# python3.7
"""Contains the running controller to save the running log."""

import os
import json
import psutil

import torch
import torch.distributed as dist

from utils.formatting_utils import format_time
from .base_controller import BaseController

__all__ = ['RunningLogger']


class RunningLogger(BaseController):
    """Defines the running controller to save the running log.

    This controller is able to save the log message in different formats:

    (1) Text format, which will be printed on screen and saved to the log file.
    (2) JSON Lines format.
    (3) TensorBoard format.

    User can use `tb_groups` attribute to group items in TensorBoard.

    Example:

        The following configuration can group two `SingleStats`
        (see `runners/running_stats.py`) `d_gan_fake` and `d_gan_real`
        into one figure, named `Loss/loss_d`, in TensorBoard.

        .. code-block:: python

            self.config.controllers.RunningLogger.update(
                dict(tb_groups={
                    'Loss/loss_d': ['d_gan_fake',
                                    'd_gan_real']
                })
            )

    NOTE:
        The controller is set to `90` priority by default.
    """

    def __init__(self, config=None):
        config = config or dict()
        config.setdefault('priority', 90)
        config.setdefault('every_n_iters', 1)
        super().__init__(config)

        self._log_order = config.get('log_order', None)
        self._log_resources = config.get('log_resources', True)
        _tb_groups = config.get('tb_groups', None)
        self._stats_name_to_tb_group_name = dict()
        if _tb_groups is not None:
            for group_name, stats_list in _tb_groups.items():
                if not isinstance(stats_list, (list, tuple)):
                    stats_list = [stats_list]
                for stats_name in stats_list:
                    self._stats_name_to_tb_group_name[stats_name] = group_name

    def setup(self, runner):
        runner.running_stats.log_order = self._log_order
        assert runner.logger is not None
        runner.logger.info('Logging settings:', indent_level=2)
        if self._log_order:
            runner.logger.info(f'Log order: {self._log_order}', indent_level=3)
        else:
            runner.logger.info('No particular log order '
                               '(first register, first logged)', indent_level=3)
        runner.logger.info(f'Log resources: {self._log_resources}',
                           indent_level=3)
        if self._stats_name_to_tb_group_name:
            runner.logger.info('TensorBoard stats grouping:', indent_level=3)
            for stats, group in self._stats_name_to_tb_group_name.items():
                runner.logger.info(f'Stats {stats} belongs to group {group}',
                                   indent_level=4)
        else:
            runner.logger.info('Each stats occupies a TensorBoard group',
                               indent_level=3)
        super().setup(runner)

    def execute_after_iteration(self, runner):
        runner.running_stats.summarize()

        # Prepare progress log.
        iter_msg = f'Iter {runner.iter:6d}/{runner.total_iters:6d}'
        if runner.seen_img < 1_000:
            iter_msg += f' ({runner.seen_img:3d} img)'
        elif runner.seen_img < 1_000_000:
            iter_msg += f' ({runner.seen_img / 1000:5.1f} K)'
        else:
            iter_msg += f' ({int(runner.seen_img / 1000 + 0.5):5d} K)'

        # Summarize process specific info.
        # Memory footprint (accessed by CPU).
        memory = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
        memory_tensor = torch.as_tensor(memory, device=runner.device)
        memory_list = []
        for _ in range(runner.world_size):
            memory_list.append(torch.zeros_like(memory_tensor))
        dist.all_gather(memory_list, memory_tensor)
        total_memory = torch.stack(memory_list, dim=0).sum().item()

        # GPU memory footprint.
        gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)

        # Reset peak GPU memory stats.
        torch.cuda.reset_peak_memory_stats()

        # Save per-rank process info in JSON Lines format.
        proc_log_path = os.path.join(runner.resource_dir,
                                     f'rank{runner.rank:02d}_proc_info.jsonl')
        with open(proc_log_path, 'a+') as proc_log:
            proc_data = {
                iter_msg: {
                    'Memory (GB)': memory,
                    'GPU Memory (GB)': gpu_memory
                }
            }
            json.dump(proc_data, proc_log)
            proc_log.write('\n')

        # Log main info via chief runner.
        if not runner.is_chief:
            return

        # Prepare log data.
        log_data = {name: stats.summarized_val
                    for name, stats in runner.running_stats.stats_pool.items()}

        # Parse log data to text format.
        msg = f'{iter_msg}, {runner.running_stats}'

        # Analyze computing resources.
        disk_free = psutil.disk_usage('.').free / (1024 ** 3)
        msg += (f'  [GPU memory: {gpu_memory:.2f} G,'
                f' Rank {runner.rank:02d} memory: {memory:.2f} G,'
                f' Total memory: {total_memory:.2f} G,'
                f' Disk free: {disk_free:.1f} G]')

        # Save overall resource info in JSON Lines format.
        proc_log_path = os.path.join(runner.resource_dir, 'all_proc_info.jsonl')
        with open(proc_log_path, 'a+') as proc_log:
            proc_data = {
                iter_msg: {
                    'Total memory (GB)': total_memory,
                    'Disk free (GB)': disk_free
                }
            }
            json.dump(proc_data, proc_log)
            proc_log.write('\n')

        # Estimate ETA.
        eta = log_data['iter time'] * (runner.total_iters - runner.iter)
        msg += f' (ETA: {format_time(eta)})'
        runner.logger.info(msg)

        # Save in JSON Lines format.
        with open(runner.log_data_path, 'a+') as f:
            json.dump(log_data, f)
            f.write('\n')

        # Save in TensorBoard format.
        if runner.tb_writer is not None:
            # Log resources.
            if self._log_resources:
                runner.tb_writer.add_scalar(
                    'Resources/GPU Mem (GB)', gpu_memory, runner.iter)
                runner.tb_writer.add_scalar(
                    'Resources/Total Mem (GB)', total_memory, runner.iter)
                for rank, rank_mem in enumerate(memory_list):
                    runner.tb_writer.add_scalar(
                        f'Resources/Rank {rank:02d} Mem (GB)', rank_mem.item(),
                        runner.iter)
                runner.tb_writer.add_scalar(
                    'Resources/Disk Free (GB)', disk_free, runner.iter)

            group_dict = dict()
            for name, stats in runner.running_stats.stats_pool.items():
                if name in ['data time', 'iter time', 'run time']:
                    continue
                if name not in self._stats_name_to_tb_group_name:
                    runner.tb_writer.add_scalar(
                        name, stats.summarized_val, runner.iter)
                else:
                    group_name = self._stats_name_to_tb_group_name[name]
                    temp_dict = group_dict.get(group_name, dict())
                    temp_dict[name] = stats.summarized_val
                    group_dict[group_name] = temp_dict

            for group_name, groupped_values in group_dict.items():
                runner.tb_writer.add_scalars(
                    group_name, groupped_values, runner.iter)

            runner.tb_writer.flush()
