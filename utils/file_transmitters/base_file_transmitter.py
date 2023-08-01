# python3.7
"""Contains the base class to transmit files across file systems.

Basically, a file transmitter connects the local file system, on which the
programme runs, to a remote file system. This is particularly used for
(1) pulling files that are required by the programme from remote, and
(2) pushing results that are produced by the programme to remote. In this way,
the programme can focus on local file system only.

NOTE: The remote file system can be the same as the local file system, since
users may want to transmit files across directories.
"""

import warnings

__all__ = ['BaseFileTransmitter']


class BaseFileTransmitter(object):
    """Defines the base file transmitter.

    A transmitter should have the following functions:

    (1) pull(): The function to pull a file/directory from remote to local.
    (2) push(): The function to push a file/directory from local to remote.
    (3) remove(): The function to remove a file/directory.
    (4) make_remote_dir(): Make directory remotely.


    To simplify, each derived class just need to implement the following helper
    functions:

    (1) download_hard(): Hard download a file/directory from remote to local.
    (2) download_soft(): Soft download a file/directory from remote to local.
        This is especially used to save space (e.g., soft link).
    (3) upload(): Upload a file/directory from local to remote.
    (4) delete(): Delete a file/directory according to given path.
    """

    def __init__(self):
        pass

    @property
    def name(self):
        """Returns the class name of the file transmitter."""
        return self.__class__.__name__

    @staticmethod
    def download_hard(src, dst):
        """Downloads (in hard mode) a file/directory from remote to local."""
        raise NotImplementedError('Should be implemented in derived class!')

    @staticmethod
    def download_soft(src, dst):
        """Downloads (in soft mode) a file/directory from local to remote."""
        raise NotImplementedError('Should be implemented in derived class!')

    @staticmethod
    def upload(src, dst):
        """Uploads a file/directory from local to remote."""
        raise NotImplementedError('Should be implemented in derived class!')

    @staticmethod
    def delete(path):
        """Deletes the given path."""
        # TODO: should we secure the path to avoid mis-removing / attacks?
        raise NotImplementedError('Should be implemented in derived class!')

    def pull(self, src, dst, hard=False):
        """Pulls a file/directory from remote to local.

        The argument `hard` is to control the download mode (hard or soft).
        For example, the hard mode may hardly copy the file while the soft mode
        may softly link the file.
        """
        if hard:
            self.download_hard(src, dst)
        else:
            self.download_soft(src, dst)

    def push(self, src, dst):
        """Pushes a file/directory from local to remote."""
        self.upload(src, dst)

    def remove(self, path):
        """Removes the given path."""
        warnings.warn(f'`{path}` will be removed!')
        self.delete(path)

    def make_remote_dir(self, directory):
        """Makes a directory on the remote system."""
        raise NotImplementedError('Should be implemented in derived class!')
