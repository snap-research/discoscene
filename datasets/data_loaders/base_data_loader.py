# python3.7
"""Contains the base data loader.

A data loader contains a dataset and an index sampler.

A dataset supports reading data from disk, parsing annotations, and
pre-processing a single sample that will be further fed into the model. Please
refer to `datasets/base_dataset.py` for more details.

The index sampler is used to maintain the order of the samples within the given
dataset. It also helps distribute all samples to different replicas evenly.
Pleaser refer to `datasets/data_loaders/distributed_sampler.py` for more
details.

Based on the dataset and the index sampler, the data loader is responsible for
grouping samples (by indices) into batches, and produces batches iteratively.
"""

__all__ = ['BaseDataLoader']


class BaseDataLoader(object):
    """Defines the base data loader.

    A data loader should contain the following members:

    (1) dataset: The dataset used to fetch data.
    (2) sampler: The index sampler used to maintain the sample order in the
        distributed environment.
    (3) iter_loader: A generator used to produce a batch of samples iteratively.

    The base data loader provides the following common functions:

    (1) __next__(): Get the next batch of data and handle the case when the
        iteration batch generator ends.
    (2) reset_batch_size(): Reset the batch size of the data produced by the
        data loader, which is particularly used for dynamic batch size
        adjustment during the training process. Basically, this function calls
        `self.build()` without re-loading the dataset. The purpose of such a
        design is that, loading a dataset (e.g., loading a huge zip file into
        memory) can be time consuming in practice.

    A derived data loader should implement the following functions:

    (1) __len__(): Return the length of the data loader, indicating how many
        batches will be produced.
    (2) build(): Set up the data loader by building the index sampler and the
        batch generator.
    (3) reset_iter_loader(): Reset the iter loader if the iteration batch
        generator ends.
    (4) info(): Information about the data loader. (optional)

    Initialization settings:

    (1) dataset: The dataset to load data from.
    (2) batch_size: The batch size of the data produced by each replica.

    Settings of the index sampler:
    (see `datasets/data_loaders/distributed_sampler.py`)

    (1) repeat: Repeating number of the entire dataset. (default: 1)
    (2) shuffle: Whether to shuffle the samples within each epoch.
        (default: True)
    (3) seed: Random seed used for shuffling. (default: 0)
    (4) drop_last_sample: Whether to drop the tailing samples that cannot be
        evenly distributed. (default: False)

    Additional settings:

    (1) drop_last_batch: Whether to drop the last incomplete batch.
        (default: True)
    (2) num_workers: Number of workers to prefetch data for each replica.
        (default: 0)
    (3) prefetch_factor: Number of samples loaded in advance by each worker.
        `N` means there will be a total of `N * num_workers` samples prefetched
        across all workers. (default: 2)
    """

    def __init__(self,
                 dataset,
                 batch_size,
                 repeat=1,
                 shuffle=True,
                 seed=0,
                 drop_last_sample=False,
                 drop_last_batch=True,
                 num_workers=0,
                 prefetch_factor=2):
        """Initializes the data loader."""
        self._dataset = dataset
        self._batch_size = batch_size

        self.repeat = repeat
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last_sample = drop_last_sample

        self.drop_last_batch = drop_last_batch
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        self._sampler = None
        self._iter_loader = None
        self.build()

    @property
    def name(self):
        """Returns the class name of the data loader."""
        return self.__class__.__name__

    @property
    def dataset(self):
        """Returns the dataset."""
        return self._dataset

    @property
    def batch_size(self):
        """Returns the batch size."""
        return self._batch_size

    @property
    def sampler(self):
        """Returns the sampler."""
        return self._sampler

    @property
    def iter_loader(self):
        """Returns the iteration batch generator."""
        return self._iter_loader

    def __next__(self):
        try:
            data = next(self._iter_loader)
        except StopIteration:
            self.reset_iter_loader()
            data = next(self._iter_loader)
        return data

    def reset_batch_size(self, batch_size=None):
        """Reset the batch size of the data produced by the data loader."""
        batch_size = batch_size or 0
        if batch_size <= 0 or batch_size == self.batch_size:
            return
        self._batch_size = batch_size
        self.build()

    def __len__(self):
        """Number of batches produced by each replica."""
        raise NotImplementedError('Should be implemented in derived class!')

    def build(self):
        """Builds data loader.

        Basically, this function sets up the index sampler and the iteration
        batch generator based on the given dataset.
        """
        raise NotImplementedError('Should be implemented in derived class!')

    def reset_iter_loader(self):
        """Resets the iteration batch generator."""
        raise NotImplementedError('Should be implemented in derived class!')

    def info(self):
        """Collects the information of the data loader.

        Please append new information in derived class if needed.
        """
        data_loader_info = {
            'Type': self.name,
            'Batch size': self.batch_size,
            'Length (num batches)': len(self),
            'Drop last incomplete batch': self.drop_last_batch,
            'Num workers for prefetching': self.num_workers,
            'Prefetch factor': self.prefetch_factor
        }
        return data_loader_info
