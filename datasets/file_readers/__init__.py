# python3.7
"""Collects all file readers."""

from .directory_reader import DirectoryReader
from .lmdb_reader import LmdbReader
from .tar_reader import TarReader
from .zip_reader import ZipReader

__all__ = ['build_file_reader']

_READERS = {
    'dir': DirectoryReader,
    'lmdb': LmdbReader,
    'tar': TarReader,
    'zip': ZipReader
}


def build_file_reader(reader_type='zip'):
    """Builds a file reader.

    Args:
        reader_type: Type of the file reader, which is case insensitive.
            (default: `zip`)

    Raises:
        ValueError: If the `reader_type` is not supported.
    """
    reader_type = reader_type.lower()
    if reader_type not in _READERS:
        raise ValueError(f'Invalid reader type: `{reader_type}`!\n'
                         f'Types allowed: {list(_READERS)}.')
    return _READERS[reader_type]
