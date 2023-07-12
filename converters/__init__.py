# python3.7
"""Collects all model converters."""

from .pggan_converter import PGGANConverter
from .stylegan_converter import StyleGANConverter
from .stylegan2_converter import StyleGAN2Converter
from .stylegan2ada_tf_converter import StyleGAN2ADATFConverter
from .stylegan2ada_pth_converter import StyleGAN2ADAPTHConverter
from .stylegan3_converter import StyleGAN3Converter

__all__ = ['build_converter']

_CONVERTERS = {
    'pggan': PGGANConverter,
    'stylegan': StyleGANConverter,
    'stylegan2': StyleGAN2Converter,
    'stylegan2ada_tf': StyleGAN2ADATFConverter,
    'stylegan2ada_pth': StyleGAN2ADAPTHConverter,
    'stylegan3': StyleGAN3Converter
}


def build_converter(model_type, verbose_log=False):
    """Builds a converter based on the model type.

    Args:
        model_type: Type of the model that the converter serves, which is case
            sensitive.
        verbose_log: Whether to print verbose log messages. (default: False)

    Raises:
        ValueError: If the `model_type` is not supported.
    """
    if model_type not in _CONVERTERS:
        raise ValueError(f'Invalid model type: `{model_type}`!\n'
                         f'Types allowed: {list(_CONVERTERS)}.')

    return _CONVERTERS[model_type](verbose_log=verbose_log)
