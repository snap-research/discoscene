# python3.7
"""Contains the running controller to control progressive training.

This controller is applicable to the models that need to progressively change
the batch size, learning rate, etc.
"""

import numpy as np

from .base_controller import BaseController

__all__ = ['ProgressSchedulerV2']

_BATCH_SIZE_SCHEDULE_DICT = {
    4: 16, 8: 8, 16: 4, 32: 2, 64: 1, 128: 1, 256: 1, 512: 1, 1024: 1
}
_MAX_BATCH_SIZE = 64

_LEARNING_RATE_SCHEDULE_DICT = {
    4: 1, 8: 1, 16: 1, 32: 1, 64: 1, 128: 1.5, 256: 2, 512: 3, 1024: 3
}


class ProgressSchedulerV2(BaseController):
    """Defines the running controller to control progressive training.

    NOTE: The controller is set to `HIGH` priority by default.
    """

    def __init__(self, config):
        assert isinstance(config, dict)
        config.setdefault('priority', 'HIGH')
        config.setdefault('every_n_iters', 1)
        super().__init__(config)

        self.base_batch_size = 0
        self.base_lrs = dict()

        self.total_img = 0
        self.init_res = config.get('init_res', 4)
        self.final_res = config.get('final_res', self.init_res)
        self.init_lod = np.log2(self.final_res // self.init_res)
        self.batch_size_schedule = config.get('batch_size_schedule', dict())
        self.lr_schedule = config.get('lr_schedule', dict())
        self.minibatch_repeats = config.get('minibatch_repeats', 4)

        self.lod_training_img = config.get('lod_training_img', [600_000,]*int(self.init_lod))
        self.lod_transition_img = config.get('lod_transition_img', [600_000,]*int(self.init_lod))
        # self.lod_duration = (self.lod_training_img + self.lod_transition_img)
        lod_durations = [lod_training_img + lod_transition_img for (lod_training_img, lod_transition_img)  in zip(self.lod_training_img, self.lod_transition_img)]
        self.lod_sum_img = []
        lod_sum = 0
        for lod_duration in lod_durations:
            lod_sum += lod_duration
            self.lod_sum_img.append(lod_sum)

        # Whether to reset the optimizer state at the beginning of each phase.
        self.reset_optimizer = config.get('reset_optimizer', True)

    def get_batch_size(self, resolution):
        """Gets batch size for a particular resolution."""
        if self.batch_size_schedule:
            return self.batch_size_schedule.get(
                f'res{resolution}', self.base_batch_size)
        batch_size_scale = _BATCH_SIZE_SCHEDULE_DICT[resolution]
        return min(_MAX_BATCH_SIZE, self.base_batch_size * batch_size_scale)

    def get_lr_scale(self, resolution):
        """Gets learning rate scale for a particular resolution."""
        if self.lr_schedule:
            return self.lr_schedule.get(f'res{resolution}', 1)
        return _LEARNING_RATE_SCHEDULE_DICT[resolution]

    def setup(self, runner):
        # Set level-of-details (lod).
        runner.lod = 0.0

        # Save default batch size and learning rate.
        self.base_batch_size = runner.batch_size
        for lr_name, lr_scheduler in runner.lr_schedulers.items():
            self.base_lrs[lr_name] = lr_scheduler.base_lrs

        # Add running stats for logging.
        runner.running_stats.add('Misc/Level of Details (lod)',
                                 log_format='4.2f',
                                 log_name='lod',
                                 log_strategy='CURRENT',
                                 requires_sync=False)
        runner.running_stats.add('Misc/Minibatch',
                                 log_format='4d',
                                 log_name='minibatch',
                                 log_strategy='CURRENT',
                                 requires_sync=False)

        # Log progressive schedule.
        runner.logger.info(
            f'Starting resolution: {self.init_res}', indent_level=2)
        runner.logger.info(
            f'Training images per lod: {self.lod_training_img}',
            indent_level=2)
        runner.logger.info(
            f'Transition images per lod: {self.lod_transition_img}',
            indent_level=2)
        runner.logger.info(
            f'Schedule is adjusted every {self.minibatch_repeats} minibatches',
            indent_level=2)
        if self.reset_optimizer:
            runner.logger.info(
                'Reset the optimizer at the beginning of each lod',
                indent_level=2)
        else:
            runner.logger.info(
                'Do not reset the optimizer at the beginning of each lod',
                indent_level=2)

        runner.logger.info('Progressive schedule:', indent_level=2)
        res = self.init_res
        lod = int(self.init_lod)
        while res <= self.final_res:
            batch_size = self.get_batch_size(res)
            lr_scale = self.get_lr_scale(res)
            runner.logger.info(
                f'Resolution {res:4d} (lod {lod}): '
                f'batch size {batch_size:3d} * {runner.world_size:2d}, '
                f'learning rate scale {lr_scale:.1f}',
                indent_level=3)
            res *= 2
            lod -= 1
        assert lod == -1 and res == self.final_res * 2
        super().setup(runner)

        # Compute total running iterations.
        self.total_img = runner.config.total_img
        current_img = 0
        num_iters = 0
        target_img = self.lod_training_img[0] 
        for phase in range(0, int(self.init_lod) - 1):
            resolution = self.init_res * (2 ** phase)
            minibatch = self.get_batch_size(resolution) * runner.world_size
            # target_img = self.lod_training_img + self.lod_duration * phase
            target_img = target_img + (self.lod_transition_img[phase] + self.lod_training_img[phase+1])
            phase_img = min(target_img, self.total_img) - current_img
            phase_iters = (phase_img + minibatch - 1) // minibatch
            if current_img + phase_img == self.total_img:
                current_img = self.total_img
                num_iters += phase_iters
                break
            if phase_iters % self.minibatch_repeats != 0:
                num_repeats = phase_iters // self.minibatch_repeats + 1
                phase_iters = num_repeats * self.minibatch_repeats
            current_img += phase_iters * minibatch
            num_iters += phase_iters
        # Last phase.
        minibatch = self.get_batch_size(self.final_res) * runner.world_size
        phase_img = self.total_img - current_img
        phase_iters = (phase_img + minibatch - 1) // minibatch
        num_iters += phase_iters
        runner.total_iters = num_iters

    def execute_before_iteration(self, runner):
        # Adjust hyper-parameters only at some particular iteration.
        is_first_iter = (runner.iter - runner.start_iter == 1)
        if (not is_first_iter) and (self.minibatch_repeats!=1 and runner.iter % self.minibatch_repeats != 1):
            return

        # Compute level-of-details.
        # phase, subphase = divmod(runner.seen_img, self.lod_duration)
        phase = 0
        for idx, lod_sum in enumerate(self.lod_sum_img):
            phase = idx
            if runner.seen_img < lod_sum:
                break
        subphase = runner.seen_img - (self.lod_sum_img[phase-1] if phase>0 else 0)
         
        # print(f'phase: {phase}, subphase:{subphase}')
        lod = self.init_lod - phase
        if self.lod_transition_img:
            transition_img = max(subphase - self.lod_training_img[phase], 0)
            lod = lod - transition_img / self.lod_transition_img[phase]
        lod = max(lod, 0.0)
        resolution = self.init_res * (2 ** int(np.ceil(self.init_lod - lod)))
        batch_size = self.get_batch_size(resolution)
        lr_scale = self.get_lr_scale(resolution)

        # Reset the batch size and adjust the learning rate if needed.
        if is_first_iter or int(lod) != int(runner.lod):
            runner.logger.info(f'Reset the batch size and '
                               f'adjust the learning rate '
                               f'at iter {runner.iter:06d} (lod {lod:.6f}).')
            runner.batch_size = batch_size
            runner.train_loader.reset_batch_size(batch_size)
            for lr_name, base_lrs in self.base_lrs.items():
                runner.lr_schedulers[lr_name].base_lrs = [
                    lr * lr_scale for lr in base_lrs]

        # Reset optimizer state if needed.
        if self.reset_optimizer:
            if (int(lod) != int(runner.lod) or
                    np.ceil(lod) != np.ceil(runner.lod)):
                runner.logger.info(f'Reset the optimizer state at '
                                   f'iter {runner.iter:06d} (lod {lod:.6f}).')
                for name in runner.optimizers:
                    runner.optimizers[name].state.clear()

        # Set level-of-details to runner.
        runner.lod = lod

    def execute_after_iteration(self, runner):
        runner.running_stats.update({'Misc/Level of Details (lod)': runner.lod})
        runner.running_stats.update({'Misc/Minibatch': runner.minibatch})
