# python3.7
"""Contains the class of TAR file reader.

Basically, a TAR file will be first extracted, under the same root directory as
the source TAR file, and with the same base name as the source TAR file. For
example, the TAR file `/home/data/test_data.tar.gz` will be extracted to
`/home/data/test_data/`. Then, this file reader degenerates into
`DirectoryReader`.

NOTE: TAR file is not recommended to use. Instead, please use ZIP file.
"""

import os.path
import shutil
import tarfile

from .base_reader import BaseReader

__all__ = ['TarReader']


class TarReader(BaseReader):
    """Defines a class to load TAR file.

    This is a static class, which is used to solve the problem that different
    data workers cannot share the same memory.
    """

    reader_cache = dict()

    @staticmethod
    def open(path):
        tar_files = TarReader.reader_cache
        if path not in tar_files:
            root_dir = os.path.dirname(path)
            base_dir = os.path.basename(path).split('.tar')[0]
            extract_dir = os.path.join(root_dir, base_dir)
            filenames = []
            with tarfile.open(path, 'r') as f:
                for member in f.getmembers():
                    if member.isfile():
                        filenames.append(member.name)
                f.extractall(extract_dir)
            file_info = {'extract_dir': extract_dir,
                         'filenames': filenames}
            tar_files[path] = file_info
        return tar_files[path]

    @staticmethod
    def close(path):
        tar_files = TarReader.reader_cache
        tar_file = tar_files.pop(path, None)
        if tar_file is not None:
            extract_dir = tar_file['extract_dir']
            shutil.rmtree(extract_dir)
            tar_file.clear()

    @staticmethod
    def open_anno_file(path, anno_filename=None):
        tar_file = TarReader.open(path)
        if not anno_filename:
            return None
        anno_path = os.path.join(tar_file['extract_dir'], anno_filename)
        if not os.path.isfile(anno_path):
            return None
        return open(anno_path, 'r')

    @staticmethod
    def _get_file_list(path):
        tar_file = TarReader.open(path)
        return tar_file['filenames']

    @staticmethod
    def fetch_file(path, filename):
        tar_file = TarReader.open(path)
        with open(os.path.join(tar_file['extract_dir'], filename), 'rb') as f:
            file_bytes = f.read()
        return file_bytes
