# python3.7
"""Contains the class of LMDB database reader.

This reader can summarize file list or fetch bytes of files inside a LMDB
database.
"""

import lmdb

from .base_reader import BaseReader

__all__ = ['LmdbReader']


class LmdbReader(BaseReader):
    """Defines a class to load LMDB file.

    This is a static class, which is used to solve the problem that different
    data workers cannot share the same memory.
    """

    reader_cache = dict()

    @staticmethod
    def open(path):
        """Opens a lmdb file."""
        lmdb_files = LmdbReader.reader_cache
        if path not in lmdb_files:
            env = lmdb.open(path,
                            max_readers=1,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            with env.begin(write=False) as txn:
                num_samples = txn.stat()['entries']
                keys = [key for key, _ in txn.cursor()]
            file_info = {'env': env,
                         'num_samples': num_samples,
                         'keys': keys}
            lmdb_files[path] = file_info
        return lmdb_files[path]

    @staticmethod
    def close(path):
        lmdb_files = LmdbReader.reader_cache
        lmdb_file = lmdb_files.pop(path, None)
        if lmdb_file is not None:
            lmdb_file['env'].close()
            lmdb_file.clear()

    @staticmethod
    def open_anno_file(path, anno_filename=None):
        # TODO: Support loading annotation file from LMDB.
        return None

    @staticmethod
    def _get_file_list(path):
        lmdb_file = LmdbReader.open(path)
        return lmdb_file['keys']

    @staticmethod
    def fetch_file(path, filename):
        lmdb_file = LmdbReader.open(path)
        env = lmdb_file['env']
        with env.begin(write=False) as txn:
            file_bytes = txn.get(filename)
        return file_bytes
