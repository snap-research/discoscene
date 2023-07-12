# python3.7
"""Contains the class of directory reader.

This reader can summarize file list or fetch bytes of files inside a directory.
"""

import os.path

from .base_reader import BaseReader

__all__ = ['DirectoryReader']


class DirectoryReader(BaseReader):
    """Defines a class to load directory."""

    @staticmethod
    def open(path):
        assert os.path.isdir(path), f'Directory `{path}` is invalid!'
        return path

    @staticmethod
    def close(path):
        _ = path  # Dummy function.

    @staticmethod
    def open_anno_file(path, anno_filename=None):
        path = DirectoryReader.open(path)
        if not anno_filename:
            return None
        anno_path = os.path.join(path, anno_filename)
        if not os.path.isfile(anno_path):
            return None
        return open(anno_path, 'r')

    @staticmethod
    def _get_file_list(path):
        path = DirectoryReader.open(path)
        return os.listdir(path)

    @staticmethod
    def fetch_file(path, filename):
        path = DirectoryReader.open(path)
        with open(os.path.join(path, filename), 'rb') as f:
            file_bytes = f.read()
        return file_bytes
