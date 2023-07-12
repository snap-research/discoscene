# python3.7
"""Contains the running controller to adjust the learning rate."""

from torch.optim import lr_scheduler

from .base_controller import BaseController
import warnings

__all__ = ['build_lr_scheduler', 'LRScheduler']


class BaseWarmUpLR(lr_scheduler._LRScheduler):  # pylint: disable=protected-access
    """Defines a base learning rate scheduler with warm-up.

    NOTE: Different from the official LRSchedulers, the base unit for learning
    rate update is always set as `iteration` instead of `epoch`. Hence, the
    number of epochs should be converted to number of iterations before using.
    """

    def __init__(self,
                 optimizer,
                 warmup_type='NO',
                 warmup_iters=0,
                 warmup_factor=0.1):
        """Initializes the scheduler with warm-up settings.

        Following warm-up types are supported:

        (1) `NO`: Do not use warm-up.
        (2) `CONST`: Use a constant value for warm-up.
        (3) `LINEAR`: Increase the learning rate linearly.
        (4) `EXP`: Increase the learning rate exponentially.

        Whatever warm-type is used, the initial learning rate for warm-up (if
        needed) is always set as `base_lr * warmup_factor`.

        Args:
            optimizer: The optimizer for applying gradients.
            warmup_type: The warm-up type. (default: `NO`)
            warmup_iters: Iterations for warm-up. (default: 0)
            warmup_factor: Factor to set the initial learning rate for warm-up.
                (default: 0.1)
        """
        self._warmup_type = warmup_type.upper()
        assert self.warmup_type in ['NO', 'CONST', 'LINEAR', 'EXP']
        self._warmup_iters = warmup_iters
        self._warmup_factor = float(warmup_factor)
        super().__init__(optimizer, last_epoch=-1)

    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False

        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_lr"):
                    values = self._get_closed_form_lr()
                else:
                    values = self.get_lr()

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group['lr'] = lr

            # if '_lr_multiplier' in param_group:
            #     param_group['lr'] = lr * param_group['_lr_multiplier']
            # else:
            #     param_group['lr'] = lr
            #     print(param_group['_name'], param_group['lr'])
            self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    @property
    def warmup_type(self):
        """Gets the warm-up type."""
        return self._warmup_type

    @property
    def warmup_iters(self):
        """Gets the iterations for warm-up."""
        return self._warmup_iters

    @property
    def warmup_factor(self):
        """Gets the warm-up factor."""
        return self._warmup_factor

    def get_warmup_lr(self):
        """Gets learning rate at the warm-up stage."""
        progress = self.last_epoch / self.warmup_iters
        if self.warmup_type == 'NO':
            return self.base_lrs
        if self.warmup_type == 'CONST':
            return [lr * self.warmup_factor for lr in self.base_lrs]
        if self.warmup_type == 'LINEAR':
            scale = (1 - progress) * (1 - self.warmup_factor)
            return [lr * (1 - scale) for lr in self.base_lrs]
        if self.warmup_type == 'EXP':
            scale = self.warmup_factor ** (1 - progress)
            return [lr * scale for lr in self.base_lrs]
        raise ValueError(f'Invalid warm-up type `{self.warmup_type}`!')

    def _get_lr(self):
        """Gets the learning rate ignoring warm-up."""
        raise NotImplementedError('Should be implemented in derived classes!')

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            return self.get_warmup_lr()
        return self._get_lr()


class FixedWarmUpLR(BaseWarmUpLR):
    """Defines a warm-up LRScheduler with fixed learning rate."""

    def _get_lr(self):
        return self.base_lrs


class StepWarmUpLR(BaseWarmUpLR):
    """Defines a warm-up LRScheduler with periodically decayed learning rate.

    In particular, the learning rate will be decayed with factor `decay_factor`
    every `decay_step` iterations.

    If the `decay_step` is a list of integers, the learning rate will be
    adjusted at those particular iterations.
    """

    def __init__(self,
                 optimizer,
                 decay_step,
                 decay_factor=0.1,
                 warmup_type='NO',
                 warmup_iters=0,
                 warmup_factor=0.1):
        self._decay_step = decay_step
        self._decay_factor = decay_factor
        super().__init__(optimizer, warmup_type, warmup_iters, warmup_factor)

    @property
    def decay_step(self):
        """Gets the decay step."""
        return self._decay_step

    @property
    def decay_factor(self):
        """Gets the decay factor."""
        return self._decay_factor

    def _get_lr(self):
        if isinstance(self.decay_step, int):
            scale = self.decay_factor ** (self.last_epoch // self.decay_step)
            return [lr * scale for lr in self.base_lrs]
        if isinstance(self.decay_step, (list, tuple)):
            bucket_id = 0
            for step in set(self.decay_step):
                if self.last_epoch >= step:
                    bucket_id += 1
            scale = self.decay_factor ** bucket_id
            return [lr * scale for lr in self.base_lrs]
        raise TypeError(f'Type of LR decay step can only be integer, list, '
                        f'or tuple, but `{type(self.decay_step)}` is received!')


class EXPStepWarmUpLR(BaseWarmUpLR):
    """Defines a warm-up LRScheduler with exponentially decayed learning rate.

    In particular, the learning rate will be decayed with factor `decay_factor`
    every `decay_step` iterations.

    If the `decay_step` is a list of integers, the learning rate will be
    adjusted at those particular iterations.
    """
    def __init__(self,
                 optimizer,
                 decay_step,
                 decay_factor=0.1,
                 warmup_type='NO',
                 warmup_iters=0,
                 warmup_factor=0.1):
        self._decay_step = decay_step
        self._decay_factor = decay_factor
        super().__init__(optimizer, warmup_type, warmup_iters, warmup_factor)

    @property
    def decay_step(self):
        """Gets the decay step."""
        return self._decay_step

    @property
    def decay_factor(self):
        """Gets the decay factor."""
        return self._decay_factor

    def _get_lr(self):
        if isinstance(self.decay_step, int):
            scale = self.decay_factor ** (self.last_epoch / self.decay_step)
            return [lr * scale for lr in self.base_lrs]
        if isinstance(self.decay_step, (list, tuple)):
            bucket_id = 0
            for step in set(self.decay_step):
                if self.last_epoch >= step:
                    bucket_id += 1
            scale = self.decay_factor ** bucket_id
            return [lr * scale for lr in self.base_lrs]
        raise TypeError(f'Type of LR decay step can only be integer, list, '
                        f'or tuple, but `{type(self.decay_step)}` is received!')


_ALLOWED_LR_TYPES = ['FIXED', 'STEP', 'EXPSTEP']


def build_lr_scheduler(config, optimizer):
    """Builds a learning rate scheduler for the given optimizer.

    Basically, the configuration is expected to contain following settings:

    (1) lr_type: The type of the learning rate scheduler. (required)
    (2) warmup_type: The warm-up type. (default: `NO`)
    (3) warmup_iters: Iterations for warm-up. (default: 0)
    (4) warmup_factor: Factor to set the initial learning rate for warm-up.
        (default: 0.1)
    (5) **kwargs: Additional settings for the scheduler.

    Args:
        config: The configuration used to build the learning rate scheduler.
        optimizer: The optimizer which the scheduler serves.

    Returns:
        A `BaseWarmUpLR` class.

    Raises:
        ValueError: The `lr_type` is not supported.
        NotImplementedError: If `lr_type` is not implemented.
    """
    assert isinstance(config, dict)
    lr_type = config['lr_type'].upper()
    warmup_type = config.get('warmup_type', 'NO')
    warmup_iters = config.get('warmup_iters', 0)
    warmup_factor = config.get('warmup_factor', 0.1)

    if lr_type not in _ALLOWED_LR_TYPES:
        raise ValueError(f'Invalid learning rate scheduler type `{lr_type}`!'
                         f'Allowed types: {_ALLOWED_LR_TYPES}.')

    if lr_type == 'FIXED':
        return FixedWarmUpLR(optimizer=optimizer,
                             warmup_type=warmup_type,
                             warmup_iters=warmup_iters,
                             warmup_factor=warmup_factor)
    if lr_type == 'STEP':
        return StepWarmUpLR(optimizer=optimizer,
                            decay_step=config['decay_step'],
                            decay_factor=config.get('decay_factor', 0.1),
                            warmup_type=warmup_type,
                            warmup_iters=warmup_iters,
                            warmup_factor=warmup_factor)
    if lr_type == 'EXPSTEP':
        return EXPStepWarmUpLR(optimizer=optimizer,
                               decay_step=config['decay_step'],
                               decay_factor=config.get('decay_factor', 0.1),
                               warmup_type=warmup_type,
                               warmup_iters=warmup_iters,
                               warmup_factor=warmup_factor)
    raise NotImplementedError(f'Not implemented scheduler type `{lr_type}`!')


class LRScheduler(BaseController):
    """Defines the running controller to adjust the learning rate.

    This controller will be executed after every iteration.

    NOTE: The controller is set to `FIRST` priority.
    """

    def __init__(self, lr_config):
        assert isinstance(lr_config, dict)
        config = {
            'priority': 'FIRST',
            'every_n_iters': 1,
        }
        super().__init__(config)
        self._lr_config = lr_config.copy()

    @property
    def lr_config(self):
        """Gets the configuration for learning rate scheduler."""
        return self._lr_config

    def setup(self, runner):
        if len(self.lr_config) == 0:
            return

        runner.logger.info('Learning rate schedule:', indent_level=2)
        for name, config in self.lr_config.items():
            if not config:
                continue
            if name in runner.lr_schedulers:
                raise ValueError(f'LR Scheduler `{name}` already existed!')
            if name not in runner.optimizers:
                raise ValueError(f'Optimizer `{name}` is missing!')
            runner.lr_schedulers[name] = build_lr_scheduler(
                config, runner.optimizers[name])
            runner.running_stats.add(f'Learning Rate/{name.capitalize()}',
                                     log_format='.2e',
                                     log_name=f'lr ({name.lower()})',
                                     log_strategy='CURRENT',
                                     requires_sync=False)
            runner.logger.info(f'Model `{name}`:', indent_level=3)
            for key, val in config.items():
                runner.logger.info(f'{key}: {val}', indent_level=4)
        super().setup(runner)

    def execute_after_iteration(self, runner):
        for name, scheduler in runner.lr_schedulers.items():
            scheduler.step()
            assert scheduler.last_epoch == runner.iter
            current_lr = runner.optimizers[name].param_groups[0]['lr']
            runner.running_stats.update(
                {f'Learning Rate/{name.capitalize()}': current_lr})
