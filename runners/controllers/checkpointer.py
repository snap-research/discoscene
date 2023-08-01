# python3.7
"""Contains the running controller to handle checkpoints."""

import os.path
from collections import deque

from .base_controller import BaseController

__all__ = ['Checkpointer']


class Checkpointer(BaseController):
    """Defines the running controller to handle checkpoints.

    This controller is used to save and load checkpoints.

    NOTE: This controller is set to `LAST` priority by default.

    Checkpoint saving settings:

    - save_running_metadata:Whether to save the running metadata, such as
        batch size, current iteration, etc. (default: True)
    - save_optimizer: Whether to save the optimizer. (default: True)
    - save_learning_rate: Whether to save the learning rate. (default: True)
    - save_loss: Whether to save the loss. (default: True)
    - save_augment: Whether to save the augmentation. (default: True)
    - save_running_stats: Whether to save the running stats. (default: False)

    Checkpoint clean-up settings:

    - keep_ckpt_num: How many recent checkpoints (besides `best_ckpt`) to keep.
        If set to -1, all checkpoints will be kept. (default: 20)
    """

    def __init__(self, config):
        assert isinstance(config, dict)
        config.setdefault('priority', 'LAST')
        super().__init__(config)

        # Checkpoint saving options.
        self._save_running_metadata = config.get('save_running_metadata', True)
        self._save_optimizer = config.get('save_optimizer', True)
        self._save_learning_rate = config.get('save_learning_rate', True)
        self._save_loss = config.get('save_loss', True)
        self._save_augment = config.get('save_augment', True)
        self._save_running_stats = config.get('save_running_stats', False)

        # Checkpoint clean-up options.
        self._keep_ckpt_num = config.get('keep_ckpt_num', 20)
        if self._keep_ckpt_num is None or self._keep_ckpt_num == 0:
            self._keep_ckpt_num = -1
        if self._keep_ckpt_num < 0:
            self.ckpt_queue = deque(maxlen=0)  # Save memory.
        else:
            self.ckpt_queue = deque(maxlen=self._keep_ckpt_num)

    def setup(self, runner):
        runner.logger.info('Saving settings:', indent_level=2)
        runner.logger.info(
            f'Saving running metadata: {self._save_running_metadata}',
            indent_level=3)
        runner.logger.info(
            f'Saving optimizer state: {self._save_optimizer}', indent_level=3)
        runner.logger.info(
            f'Saving learning rate scheduler: {self._save_learning_rate}',
            indent_level=3)
        runner.logger.info(
            f'Saving loss: {self._save_loss}', indent_level=3)
        runner.logger.info(
            f'Saving augment: {self._save_augment}', indent_level=3)
        runner.logger.info(
            f'Saving running stats: {self._save_running_stats}',
            indent_level=3)
        if self.ckpt_queue.maxlen > 0:
            runner.logger.info(
                f'Keep at most {self._keep_ckpt_num} checkpoints',
                indent_level=3)
        else:
            runner.logger.info('Keep all checkpoints', indent_level=3)
        super().setup(runner)

    def require_clean_up(self, runner):
        """Returns whether the outdated checkpoint should be removed."""
        if not runner.is_chief:
            assert len(self.ckpt_queue) == 0
            return False
        if self.ckpt_queue.maxlen == 0:
            return False
        return len(self.ckpt_queue) == self.ckpt_queue.maxlen

    def clean_up(self, runner):
        """Removes the outdated checkpoint."""
        if not runner.is_chief:  # Only the chief executes clean-up.
            return

        filepath = self.ckpt_queue.popleft()  # Pop out the outdated checkpoint.
        if os.path.isfile(filepath):
            # May have already been deleted by other controllers.
            runner.logger.info(f'Remove {filepath} in the routine clean-up '
                               f'of {self.name}.')
            os.remove(filepath)

    def save(self, runner):
        """Saves a checkpoint."""
        runner.check_ddp_consistency()  # Ensure consistency across replicas.
        if not runner.is_chief:  # Only the chief will save the checkpoint.
            return

        save_name = f'checkpoint-{runner.iter:06d}.pth'
        filepath = os.path.join(runner.checkpoint_dir, save_name)
        runner.save(filepath=filepath,
                    running_metadata=self._save_running_metadata,
                    optimizer=self._save_optimizer,
                    learning_rate=self._save_learning_rate,
                    loss=self._save_loss,
                    augment=self._save_augment,
                    running_stats=self._save_running_stats)
        self.ckpt_queue.append(filepath)

    def execute_after_iteration(self, runner):
        if self.require_clean_up(runner):
            self.clean_up(runner)
        self.save(runner)
