# python3.7
"""Contains the class of dummy file transmitter.

This file transmitter has all expected data transmission functions but behaves
silently, which is very useful in multi-processing mode. Only the chief process
can have the file transmitter with normal behavior.
"""

from .base_file_transmitter import BaseFileTransmitter

__all__ = ['DummyFileTransmitter']


class DummyFileTransmitter(BaseFileTransmitter):
    """Implements a dummy transmitter which transmits nothing."""

    @staticmethod
    def download_hard(src, dst):
        return

    @staticmethod
    def download_soft(src, dst):
        return

    @staticmethod
    def upload(src, dst):
        return

    @staticmethod
    def delete(path):
        return

    def make_remote_dir(self, directory):
        return
