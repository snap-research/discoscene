# python3.7
"""Contains the base class to read files.

A file reader reads data from a given file and cache the file if possible.
Typically, file readers are designed to read files from zip, lmdb, or directory.
"""

from utils.misc import IMAGE_EXTENSIONS
from utils.misc import check_file_ext

__all__ = ['BaseReader']


class BaseReader(object):
    """Defines the base file reader.

    A reader should have the following functions:

    (1) open(): The function to open (and cache) a given file/directory.
    (2) close(): The function to close a given file/directory.
    (3) open_anno_file(): The function to open a specific annotation file inside
        the given file/directory.
    (4) get_file_list(): The function to get the list of all files inside the
        given file/directory. The returned list is already sorted.
    (5) get_file_list_with_ext(): The function to get the list of files with
        expected file extensions inside the given file/directory. The returned
        list is already sorted.
    (6) get_image_list(): The function to get the list of all images inside the
        given file/directory. The returned list is already sorted.
    (7) fetch_file(): The function to fetch the bytes of member file inside the
        given file/directory.
    """

    @staticmethod
    def open(path):
        """Opens the given path."""
        raise NotImplementedError('Should be implemented in derived class!')

    @staticmethod
    def close(path):
        """Closes the given path."""
        raise NotImplementedError('Should be implemented in derived class!')

    @staticmethod
    def open_anno_file(path, anno_filename=None):
        """Opens the annotation file in `path` and returns a file pointer.

        If the annotation file does not exist, return `None`.
        """
        raise NotImplementedError('Should be implemented in derived class!')

    @staticmethod
    def _get_file_list(path):
        """Gets the list of all files inside `path`."""
        raise NotImplementedError('Should be implemented in derived class!')

    @classmethod
    def get_file_list(cls, path):
        """Gets the sorted list of all files inside `path`."""
        return sorted(cls._get_file_list(path))

    @classmethod
    def get_file_list_with_ext(cls, path, ext=None):
        """Gets the sorted list of files with expected extensions."""
        ext = ext or []
        return [f for f in cls.get_file_list(path) if check_file_ext(f, *ext)]

    @classmethod
    def get_image_list(cls, path):
        """Gets the sorted list of image files inside `path`."""
        return cls.get_file_list_with_ext(path, IMAGE_EXTENSIONS)

    @staticmethod
    def fetch_file(path, filename):
        """Fetches the bytes of file `filename` inside `path`.

        Example:

        >>> f = BaseReader.fetch_file('data', 'face.obj')
        >>> obj = f.decode('utf-8')  # convert `bytes` to `str`
        """
        raise NotImplementedError('Should be implemented in derived class!')
