# python3.7
"""Contains the running controller to control augmentation strength."""

import numpy as np

import torch

from ..utils.running_stats import SingleStats
from ..augmentations.ada_aug import AdaAug
from .base_controller import BaseController

__all__ = ['AdaAugController']


class AdaAugController(BaseController):
    """Defines the running controller to adjust the strength of augmentations.

    This controller works together with the augmentation pipeline introduces by
    StyleGAN2-ADA (https://arxiv.org/pdf/2006.06676.pdf). Concretely, AadAug,
    which is defined in `runners/augmentations/ada_aug.py`, augments the data
    based on an adjustable probability. This controller controls how this
    probability is adjusted.

    NOTE: The controller is set to `FIRST` priority.

    Basically, the aug_config is expected to contain following settings:

    (1) init_p: The init prob of augmentations. (default: 0.0)
    (2) target_p: The target (final) prob of augmentations. (default: 0.6)
    (3) every_n_iters: How often to adjust the probability. (default: 4)
    (4) speed_img: Speed to adjust the probability, which is measured in number
        of images it takes for the probability to increase/decrease by one unit.
        (default: 500_000)
    (5) strategy: The strategy to adjust the probability. Support `fixed`,
        `linear`, and `adaptive`. (default: `adaptive`)
    """

    def __init__(self, config):
        assert isinstance(config, dict)
        config.setdefault('priority', 'First')
        config.setdefault('every_n_iters', 4)
        config.setdefault('first_iter', False)
        super().__init__(config)

        self._init_p = config.get('init_p', 0.0)
        self._target_p = config.get('target_p', 0.6)
        self._speed_img = config.get('speed_img', 500_000)
        self._strategy = config.get('strategy', 'adaptive').lower()
        self._milestone = config.get('milestone', None)
        assert self._strategy in ['fixed', 'linear', 'adaptive']

    def setup(self, runner):
        """Sets the initial augmentation strength before training."""
        if not isinstance(runner.augment, AdaAug):
            raise ValueError(f'`{self.name}` only works together with '
                             f'adaptive augmentation pipeline `AdaAug`!\n')

        if self._strategy == 'fixed':
            aug_prob = self._target_p
        else:
            aug_prob = self._init_p
        runner.augment.p = torch.as_tensor(aug_prob).relu()
        runner.augment.prob_tracker = SingleStats('Aug Prob Tracker',
                                                  log_format=None,
                                                  log_strategy='AVERAGE',
                                                  requires_sync=True)
        runner.running_stats.add('Misc/Aug Prob',
                                 log_name='aug_prob',
                                 log_format='.3f',
                                 log_strategy='CURRENT',
                                 requires_sync=False,
                                 keep_previous=True)

        runner.logger.info('Adaptive augmentation settings:', indent_level=2)
        runner.logger.info(f'Strategy: {self._strategy}', indent_level=3)
        runner.logger.info(f'Initial probability: {self._init_p}',
                           indent_level=3)
        runner.logger.info(f'Target probability : {self._target_p}',
                           indent_level=3)
        runner.logger.info(f'Adjustment speed {self._speed_img} images',
                           indent_level=3)
        super().setup(runner)

    def execute_after_iteration(self, runner):
        if self._strategy == 'fixed':
            aug_prob = self._target_p
        elif self._strategy == 'linear':
            if self._milestone is not None:
                slope = min(runner.iter / self._milestone, 1)
            else:
                slope = min(runner.iter / runner.total_iters, 1)
            aug_prob = self._init_p + (self._target_p - self._init_p) * slope
        else:
            minibatch = runner.batch_size * runner.world_size
            slope = (minibatch * self.every_n_iters) / self._speed_img
            criterion = runner.augment.prob_tracker.summarize()
            adjust = np.sign(criterion - self._target_p) * slope
            aug_prob = runner.augment.p + adjust

        runner.augment.p = torch.as_tensor(aug_prob).relu()
        runner.running_stats.update({'Misc/Aug Prob': runner.augment.p})
