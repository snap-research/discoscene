# python3.7
"""Collects all augmentation pipelines."""

from .no_aug import NoAug
from .ada_aug import AdaAug

__all__ = ['build_aug']

_AUGMENTATIONS = {
    'NoAug': NoAug,
    'AdaAug': AdaAug
}


def build_aug(aug_type, **kwargs):
    """Builds a differentiable augmentation pipeline based on its class type.

    Args:
        aug_type: Class type to which the augmentation belongs, which is case
            sensitive.
        **kwargs: Additional arguments to build the aug.

    Raises:
        ValueError: If the `aug_type` is not supported.
    """
    if aug_type not in _AUGMENTATIONS:
        raise ValueError(f'Invalid augmentation type: `{aug_type}`!\n'
                         f'Types allowed: {list(_AUGMENTATIONS)}.')
    return _AUGMENTATIONS[aug_type](**kwargs)
