# python3.7
"""Contains the function to build optimizer for a model."""

import torch

__all__ = ['build_optimizer']

_ALLOWED_OPT_TYPES = ['sgd', 'adam']


def build_optimizer(config, model):
    """Builds an optimizer for the given model.

    Basically, the configuration is expected to contain following settings:

    (1) opt_type: The type of the optimizer. (required)
    (2) base_lr: The base learning rate for all parameters. (required)
    (3) base_wd: The base weight decay for all parameters. (default: 0.0)
    (4) bias_lr_multiplier: The learning rate multiplier for bias parameters.
        (default: 1.0)
    (5) bias_wd_multiplier: The weight decay multiplier for bias parameters.
        (default: 1.0)
    (6) **kwargs: Additional settings for the optimizer, such as `momentum`.

    Args:
        config: The configuration used to build the optimizer.
        model: The model which the optimizer serves.

    Returns:
        A `torch.optim.Optimizer`.

    Raises:
        ValueError: The `opt_type` is not supported.
        NotImplementedError: If `opt_type` is not implemented.
    """
    assert isinstance(config, dict)
    opt_type = config['opt_type'].lower()
    base_lr = config['base_lr']
    base_wd = config.get('base_wd', 0.0)
    bias_lr_multiplier = config.get('bias_lr_multiplier', 1.0)
    bias_wd_multiplier = config.get('bias_wd_multiplier', 1.0)
    custom_keys = config.get('custom_keys', None)

    if opt_type not in _ALLOWED_OPT_TYPES:
        raise ValueError(f'Invalid optimizer type `{opt_type}`!'
                         f'Allowed types: {_ALLOWED_OPT_TYPES}.')

    if custom_keys is not None:
        for key in custom_keys:
            assert 'lr_multiplier' in custom_keys[key], 'Lr_mult must be included by the "custom_keys"'

    model_params = []
    for param_name, param in model.named_parameters():
        param_group = {'params': [param]}
        is_custom = False
        if custom_keys is not None:
            for key in custom_keys:
                if param_name.startswith(key):
                    is_custom = True
                    param_group['lr'] = base_lr * custom_keys[key]['lr_multiplier']
                    param_group['weight_decay'] = base_wd * custom_keys[key].get('wd_multiplier', 1.0)
                    param_group['_lr_multiplier'] = custom_keys[key]['lr_multiplier']
                    # param_group['_wd_multiplier'] = custom_keys[key].get('wd_multiplier', 1.0)
                    param_group['_name'] = param_name
                    if 'beta1_multiplier' in  custom_keys[key] or 'beta2_multiplier' in  custom_keys[key]:
                        ori_betas = list(config.get('betas', (0.9, 0.999)))
                        ori_betas[0] = ori_betas[0] * custom_keys[key].get('beta1_multiplier', 1)
                        ori_betas[1] = ori_betas[1] * custom_keys[key].get('beta2_multiplier', 1)
                        param_group['betas'] = tuple(ori_betas)
        if not is_custom:
            if 'bias' in param_name:
                param_group['lr'] = base_lr * bias_lr_multiplier
                param_group['weight_decay'] = base_wd * bias_wd_multiplier
            else:
                param_group['lr'] = base_lr
                param_group['weight_decay'] = base_wd
            param_group['_name'] = param_name
        model_params.append(param_group)

    if opt_type == 'sgd':
        return torch.optim.SGD(params=model_params,
                               lr=base_lr,
                               momentum=config.get('momentum', 0.9),
                               dampening=config.get('dampening', 0),
                               weight_decay=base_wd,
                               nesterov=config.get('nesterov', False))
    if opt_type == 'adam':
        opt = torch.optim.Adam(params=model_params,
                                lr=base_lr,
                                betas=config.get('betas', (0.9, 0.999)),
                                eps=config.get('eps', 1e-8),
                                weight_decay=base_wd,
                                amsgrad=config.get('amsgrad', False))
        return opt
    raise NotImplementedError(f'Not implemented optimizer type `{opt_type}`!')
