# python3.7
"""Collects all file transmitters."""

from .local_file_transmitter import LocalFileTransmitter
from .dummy_file_transmitter import DummyFileTransmitter

__all__ = ['build_file_transmitter']

_TRANSMITTERS = {
    'local': LocalFileTransmitter,
    'dummy': DummyFileTransmitter,
}


def build_file_transmitter(transmitter_type='local', **kwargs):
    """Builds a file transmitter.

    Args:
        transmitter_type: Type of the file transmitter_type, which is case
            insensitive. (default: `normal`)
        **kwargs: Additional arguments to build the file transmitter.

    Raises:
        ValueError: If the `transmitter_type` is not supported.
    """
    transmitter_type = transmitter_type.lower()
    if transmitter_type not in _TRANSMITTERS:
        raise ValueError(f'Invalid transmitter type: `{transmitter_type}`!\n'
                         f'Types allowed: {list(_TRANSMITTERS)}.')
    return _TRANSMITTERS[transmitter_type](**kwargs)
