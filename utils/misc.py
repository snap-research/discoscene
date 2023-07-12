# python3.7
"""Misc utility functions."""

import os
import hashlib

from torch.hub import download_url_to_file
import torch
import numpy as np

__all__ = [
    'REPO_NAME', 'Infix', 'print_and_execute', 'check_file_ext',
    'IMAGE_EXTENSIONS', 'VIDEO_EXTENSIONS', 'MEDIA_EXTENSIONS', 'parse_ann_format',
    'parse_file_format', 'set_cache_dir', 'get_cache_dir', 'download_url', 'gather_data'
]

REPO_NAME = 'Hammer'  # Name of the repository (project).


class Infix(object):
    """Helper class to create custom infix operators.

    When using it, make sure to put the operator between `<<` and `>>`.
    `<< INFIX_OP_NAME >>` should be considered as a whole operator.

    Examples:

    # Use `Infix` to create infix operators directly.
    add = Infix(lambda a, b: a + b)
    1 << add >> 2  # gives 3
    1 << add >> 2 << add >> 3  # gives 6

    # Use `Infix` as a decorator.
    @Infix
    def mul(a, b):
        return a * b
    2 << mul >> 4  # gives 8
    2 << mul >> 3 << mul >> 7  # gives 42
    """

    def __init__(self, function):
        self.function = function
        self.left_value = None

    def __rlshift__(self, left_value):  # override `<<` before `Infix` instance
        assert self.left_value is None  # make sure left is only called once
        self.left_value = left_value
        return self

    def __rshift__(self, right_value):  # override `>>` after `Infix` instance
        result = self.function(self.left_value, right_value)
        self.left_value = None  # reset to None
        return result


def print_and_execute(cmd):
    """Prints and executes a system command.

    Args:
        cmd: Command to be executed.
    """
    print(cmd)
    os.system(cmd)


def check_file_ext(filename, *ext_list):
    """Checks whether the given filename is with target extension(s).

    NOTE: If `ext_list` is empty, this function will always return `False`.

    Args:
        filename: Filename to check.
        *ext_list: A list of extensions.

    Returns:
        `True` if the filename is with one of extensions in `ext_list`,
        otherwise `False`.
    """
    if len(ext_list) == 0:
        return False
    ext_list = [ext if ext.startswith('.') else '.' + ext for ext in ext_list]
    ext_list = [ext.lower() for ext in ext_list]
    basename = os.path.basename(filename)
    ext = os.path.splitext(basename)[1].lower()
    return ext in ext_list


# File extensions regarding images (not including GIFs).
IMAGE_EXTENSIONS = (
    '.bmp', '.ppm', '.pgm', '.jpeg', '.jpg', '.jpe', '.jp2', '.png', '.webp',
    '.tiff', '.tif'
)
# File extensions regarding videos.
VIDEO_EXTENSIONS = (
    '.avi', '.mkv', '.mp4', '.m4v', '.mov', '.webm', '.flv', '.rmvb', '.rm',
    '.3gp'
)
# File extensions regarding media, i.e., images, videos, GIFs.
MEDIA_EXTENSIONS = ('.gif', *IMAGE_EXTENSIONS, *VIDEO_EXTENSIONS)


def parse_ann_format(path):
    if path is None:
        return None
    if os.path.isfile(path) and os.path.splitext(path)[1] == '':
        return 'txt'
    path = path.lower()
    if path.endswith('.txt'):  # Cannot parse accurate extension.
        return 'txt'
    ext = os.path.splitext(path)[1]
    if ext == '.json':
        return 'json'
    if ext == '.pkl':
        return 'pkl'

def parse_file_format(path):
    """Parses the file format of a given path.

    This function basically parses the file format according to its extension.
    It will also return `dir` is the given path is a directory.

    Parable file formats:

    - zip: with `.zip` extension.
    - tar: with `.tar` / `.tgz` / `.tar.gz` extension.
    - lmdb: a folder ending with `lmdb`.
    - txt: with `.txt` / `.text` extension, OR without extension (e.g. LICENSE).
    - json: with `.json` extension.
    - jpg: with `.jpeg` / `jpg` / `jpe` extension.
    - png: with `.png` extension.

    Args:
        path: The path to the file to parse format from.

    Returns:
        A lower-case string, indicating the file format, or `None` if the format
            cannot be successfully parsed.
    """
    # Handle directory.
    if os.path.isdir(path) or path.endswith('/'):
        if path.rstrip('/').lower().endswith('lmdb'):
            return 'lmdb'
        return 'dir'
    # Handle file.
    if os.path.isfile(path) and os.path.splitext(path)[1] == '':
        return 'txt'
    path = path.lower()
    if path.endswith('.tar.gz'):  # Cannot parse accurate extension.
        return 'tar'
    ext = os.path.splitext(path)[1]
    if ext == '.zip':
        return 'zip'
    if ext in ['.tar', '.tgz']:
        return 'tar'
    if ext in ['.txt', '.text']:
        return 'txt'
    if ext == '.json':
        return 'json'
    if ext in ['.jpeg', '.jpg', '.jpe']:
        return 'jpg'
    if ext == '.png':
        return 'png'
    # Unparsable.
    return None


_cache_dir = None


def set_cache_dir(directory=None):
    """Sets the global cache directory.

    The cache directory can be used to save some files that will be shared
    across jobs. The default cache directory is set as `~/.cache/${REPO_NAME}/`.
    This function can be used to redirect the cache directory. Or, users can use
    `None` to reset the cache directory back to default.

    Args:
        directory: The target directory used to cache files. If set as `None`,
            the cache directory will be reset back to default. (default: None)
    """
    assert directory is None or isinstance(directory, str), 'Invalid directory!'
    global _cache_dir  # pylint: disable=global-statement
    _cache_dir = directory


def get_cache_dir():
    """Gets the global cache directory.

    The global cache directory is primarily set as `~/.cache/${REPO_NAME}/` by
    default, and can be redirected with `set_cache_dir()`.

    Returns:
        A string, representing the global cache directory.
    """
    if _cache_dir is None:
        home = os.path.expanduser('~')
        return os.path.join(home, '.cache', REPO_NAME)
    return _cache_dir


def download_url(url, path=None, filename=None, sha256=None):
    """Downloads file from URL.

    This function downloads a file from given URL, and executes Hash check if
    needed.

    Args:
        url: The URL to download file from.
        path: Path (directory) to save the downloaded file. If set as `None`,
            the cache directory will be used. Please see `get_cache_dir()` for
            more details. (default: None)
        filename: The name to save the file. If set as `None`, this name will be
            automatically parsed from the given URL. (default: None)
        sha256: The expected sha256 of the downloaded file. If set as `None`,
            the hash check will be skipped. Otherwise, this function will check
            whether the sha256 of the downloaded file matches this field.

    Returns:
        A two-element tuple, where the first term is the full path of the
            downloaded file, and the second term indicate the hash check result.
            `True` means hash check passes, `False` means hash check fails,
            while `None` means no hash check is executed.
    """
    # Handle file path.
    if path is None:
        path = get_cache_dir()
    if filename is None:
        filename = os.path.basename(url)
    save_path = os.path.join(path, filename)
    # Download file if needed.
    if not os.path.exists(save_path):
        print(f'Downloading URL `{url}` to path `{save_path}` ...')
        os.makedirs(path, exist_ok=True)
        download_url_to_file(url, save_path, hash_prefix=None, progress=True)
    # Check hash if needed.
    check_result = None
    if sha256 is not None:
        with open(save_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read())
            check_result = (file_hash.hexdigest() == sha256)

    return save_path, check_result

def gather_data(data_list, device):
    if not isinstance(data_list, (list, tuple)):
        raise NotImplementedError
    if isinstance(data_list[0], (torch.Tensor, np.ndarray)):
        data = torch.from_numpy(np.stack(data_list)).pin_memory().to(device)
    elif isinstance(data_list[0], (dict, )):
        data = {}
        for key in data_list[0]:
            data_sub_list = [x[key] for x in data_list]
            data[key] = torch.from_numpy(np.stack(data_sub_list)).pin_memory().to(device)
    else:
        raise NotImplementedError
    return data
