# python3.7
"""Contains the class of local file transmitter.

The transmitter builds the connection between the local file system and itself.
This can be used to transmit files from one directory to another. Consequently,
`remote` in this file also means `local`.
"""

from utils.misc import print_and_execute
from .base_file_transmitter import BaseFileTransmitter

__all__ = ['LocalFileTransmitter']


class LocalFileTransmitter(BaseFileTransmitter):
    """Implements the transmitter connecting local file system to itself."""

    @staticmethod
    def download_hard(src, dst):
        print_and_execute(f'cp {src} {dst}')

    @staticmethod
    def download_soft(src, dst):
        print_and_execute(f'ln -s {src} {dst}')

    @staticmethod
    def upload(src, dst):
        print_and_execute(f'cp {src} {dst}')

    @staticmethod
    def delete(path):
        print_and_execute(f'rm -r {path}')

    def make_remote_dir(self, directory):
        print_and_execute(f'mkdir -p {directory}')
