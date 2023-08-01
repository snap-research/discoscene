# python3.7
"""Contains the class of ZIP file reader.

This reader can summarize file list or fetch bytes of files inside a ZIP.
"""

import zipfile

from .base_reader import BaseReader

__all__ = ['ZipReader']


class ZipReader(BaseReader):
    """Defines a class to load ZIP file.

    This is a static class, which is used to solve the problem that different
    data workers cannot share the same memory.
    """

    reader_cache = dict()

    @staticmethod
    def open(path):
        zip_files = ZipReader.reader_cache
        if path not in zip_files:
            zip_files[path] = zipfile.ZipFile(path, 'r')
        return zip_files[path]

    @staticmethod
    def close(path):
        zip_files = ZipReader.reader_cache
        zip_file = zip_files.pop(path, None)
        if zip_file is not None:
            zip_file.close()

    @staticmethod
    def open_anno_file(path, anno_filename=None):
        zip_file = ZipReader.open(path)
        if not anno_filename:
            return None
        if anno_filename not in zip_file.namelist():
            return None
        return zip_file.open(anno_filename, 'r')

    @staticmethod
    def _get_file_list(path):
        zip_file = ZipReader.open(path)
        return zip_file.namelist()

    @staticmethod
    def fetch_file(path, filename):
        zip_file = ZipReader.open(path)
        return zip_file.read(filename)
