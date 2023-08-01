# python3.7
"""Collects all configs."""

from .stylegan_config import StyleGANConfig
from .stylegan2_config import StyleGAN2Config
from .stylegan2_finetune_config import StyleGAN2FineTuneConfig
from .stylegan3_config import StyleGAN3Config
from .volumegan_ffhq_config import VolumeGANFFHQConfig
from .discoscene_config import DiscoSceneConfig

__all__ = ['CONFIG_POOL', 'build_config']

CONFIG_POOL = [
    StyleGANConfig,
    StyleGAN2Config,
    StyleGAN2FineTuneConfig, 
    StyleGAN3Config,
    VolumeGANFFHQConfig,
    DiscoSceneConfig,
]


def build_config(invoked_command, kwargs):
    """Builds a configuration based on the invoked command.

    Args:
        invoked_command: The command that is invoked.
        kwargs: Keyword arguments passed from command line, which will be used
            to build the configuration.

    Raises:
        ValueError: If the `invoked_command` is missing.
    """
    for config in CONFIG_POOL:
        if config.name == invoked_command:
            return config(kwargs)
    raise ValueError(f'Invoked command `{invoked_command}` is missing!\n')
