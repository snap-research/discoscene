# python3.7
"""Contains the running controller for evaluation."""

import os
import time

from utils.formatting_utils import format_time
from .base_controller import BaseController

__all__ = ['Evaluator']


class Evaluator(BaseController):
    """Defines the running controller for evaluation.

    This controller maintains all of the evaluations that should be executed in
    the training process. Each evaluation can have a different evaluation
    interval, therefore, this evaluator goes through the metric list after every
    iteration to check whether an evaluation is needed.

    NOTE: The controller is set to `LAST` priority by default.

    Best checkpoint saving settings:

    - save_running_metadata: Whether to save the running metadata, such as
        batch size, current iteration, etc. (default: True)
    - save_optimizer: Whether to save the optimizer. (default: True)
    - save_learning_rate: Whether to save the learning rate. (default: True)
    - save_loss: Whether to save the loss. (default: True)
    - save_augment: Whether to save the augmentation. (default: True)
    - save_running_stats: Whether to save the running stats. (default: False)
    """

    def __init__(self, config):
        assert isinstance(config, dict)
        config.setdefault('every_n_iters', 1)
        config.setdefault('priority', 'LAST')
        super().__init__(config)

        self.default_eval_at_start = config.get('default_eval_at_start', True)
        self.default_eval_interval = config.get('default_eval_interval', -1)

        # Best checkpoints saving options
        self.default_save_best_ckpt = config.get('default_save_best_ckpt', True)
        self.default_save_running_metadata = config.get(
            'default_save_running_metadata', True)
        self.default_save_optimizer = config.get('default_save_optimizer', True)
        self.default_save_learning_rate = config.get(
            'default_save_learning_rate', True)
        self.default_save_loss = config.get('default_save_loss', True)
        self.default_save_augment = config.get('default_save_augment', True)
        self.default_save_running_stats = config.get(
            'default_save_running_stats', False)

    def setup(self, runner):
        runner.logger.info('Evaluation strategy:', indent_level=2)
        runner.logger.info(
            f'Default evaluation at start: {self.default_eval_at_start}',
            indent_level=3)
        runner.logger.info(
            f'Default evaluation interval: {self.default_eval_interval}',
            indent_level=3)
        runner.logger.info('Saving best checkpoint strategy:', indent_level=2)
        runner.logger.info(f'Default save best checkpoint: '
                           f'{self.default_save_best_ckpt}',
                           indent_level=3)
        runner.logger.info(f'Default save running metadata: '
                           f'{self.default_save_running_metadata}',
                           indent_level=3)
        runner.logger.info(f'Default save optimizer state: '
                           f'{self.default_save_optimizer}',
                           indent_level=3)
        runner.logger.info(f'Default save learning rate scheduler: '
                           f'{self.default_save_learning_rate}',
                           indent_level=3)
        runner.logger.info(f'Default save loss: {self.default_save_loss}',
                           indent_level=3)
        runner.logger.info(f'Default save augment: {self.default_save_augment}',
                           indent_level=3)
        runner.logger.info(f'Default save running stats: '
                           f'{self.default_save_running_stats}',
                           indent_level=3)
        super().setup(runner)

    def do_evaluation(self, runner, metric):
        """Determines whether a metric should be evaluated."""
        interval = (self.default_eval_interval
            if metric['interval'] is None else metric['interval'])
        first_iter = (self.default_eval_at_start
            if metric['first_iter'] is None else metric['first_iter'])

        # Check first iteration.
        if first_iter and runner.iter - runner.start_iter == 1:
            return True

        # Always evaluated on the last iteration.
        if runner.iter == runner.total_iters:
            return True

        # Check iter-based interval.
        if interval > 0 and runner.iter % interval == 0:
            return True

        return False

    def save_best_ckpt(self, runner, metric, tag):
        """Save best checkpoint.

        Unlike checkpointer, which saves checkpoints periodically, this method
        enables evaluator to save a checkpoint right after its evaluation.
        It ensures the checkpoint is the best one (under a given metric) along
        the evaluations.

        Args:
            runner: an instance of `Runner`.
            metric: an instance of `Metric` for this checkpoint.
            tag: an identifier to save best checkpoint.

        NOTE: `runner.check_ddp_consistency()` is omitted, otherwise, checking
        DDP consistency in the chief process only will cause deadlock.
        """
        save_best = (self.default_save_best_ckpt
                     if metric['save_best'] is None else metric['save_best'])

        if not save_best:
            return
        if not runner.is_chief:
            return

        save_running_metadata = (self.default_save_running_metadata
                                 if metric['save_running_metadata'] is None
                                 else metric['save_running_metadata'])
        save_optimizer = (self.default_save_optimizer
                          if metric['save_optimizer'] is None
                          else metric['save_optimizer'])
        save_learning_rate = (self.default_save_learning_rate
                              if metric['save_learning_rate'] is None
                              else metric['save_learning_rate'])
        save_loss = (self.default_save_loss
                     if metric['save_loss'] is None else metric['save_loss'])
        save_augment = (self.default_save_augment
                        if metric['save_augment'] is None
                        else metric['save_augment'])
        save_running_stats = (self.default_save_running_stats
                              if metric['save_running_stats'] is None
                              else metric['save_running_stats'])

        for filename in os.listdir(runner.checkpoint_dir):
            if f'best-{tag}-checkpoint' in filename:
                os.remove(os.path.join(runner.checkpoint_dir, filename))
        save_name = f'best-{tag}-checkpoint-{runner.iter:06d}.pth'
        filepath = os.path.join(runner.checkpoint_dir, save_name)
        runner.save(filepath=filepath,
                    running_metadata=save_running_metadata,
                    optimizer=save_optimizer,
                    learning_rate=save_learning_rate,
                    loss=save_loss,
                    augment=save_augment,
                    running_stats=save_running_stats)

    def execute_after_iteration(self, runner):
        for metric_name, metric in runner.metrics.items():
            # Check executable.
            if not self.do_evaluation(runner, metric):
                continue

            # Set evaluation arguments.
            eval_args = [runner.val_loader]
            for model_name, model_kwargs in metric['kwargs'].items():
                eval_args.append(runner.models[model_name])
                if model_kwargs is None:
                    eval_args.append(runner.model_kwargs_val[model_name])
                else:
                    assert isinstance(model_kwargs, dict)
                    eval_args.append(model_kwargs)

            # Start evaluation.
            start_time = time.time()
            eval_result = metric['fn'].evaluate(*eval_args)
            duration_str = format_time(time.time() - start_time)

            # Save results and print log information.
            suffix = f'at iter {runner.iter:06d}'
            if runner.seen_img < 1_000:
                suffix += f' ({runner.seen_img:3d} img).'
            elif runner.seen_img < 1_000_000:
                suffix += f' ({runner.seen_img / 1000:5.1f} K).'
            else:
                suffix += f' ({int(runner.seen_img / 1000 + 0.5):5d} K).'
            suffix += f' ({duration_str})'
            metric['fn'].save(
                result=eval_result,
                target_filename=f'{metric_name}-{runner.iter:06d}',
                log_suffix=suffix,
                tag=runner.iter)

            # Save metrics for best model selection.
            if not runner.is_chief:
                # Only chief records the evaluation results.
                continue
            for key, val in eval_result.items():
                if not isinstance(val, (int, float)):
                    # Skip non-numerical metrics.
                    continue
                if key not in runner.eval_results:
                    # A temp variable used to determine whether the metric can
                    # be used to compare performance.
                    temp = 0 << metric['fn'].is_better_than(key) >> 0
                    if temp is None:
                        runner.eval_results[key] = dict()
                    else:
                        runner.eval_results[key] = {'best': (None, runner.iter)}
                runner.eval_results[key][runner.iter] = val
                if 'best' not in runner.eval_results[key]:
                    continue
                current_best = runner.eval_results[key]['best'][0]
                if val << metric['fn'].is_better_than(key) >> current_best:
                    runner.eval_results[key]['best'] = (val, runner.iter)
                    self.save_best_ckpt(runner, metric=metric, tag=key)
