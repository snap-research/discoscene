# python3.7
"""Collects all data loaders."""

import warnings

from .iter_data_loader import IterDataLoader
try:
    from .dali_data_loader import DALIDataLoader
except ImportError:
    DALIDataLoader = None

__all__ = ['build_data_loader']

_DATA_LOADERS_ALLOWED = ['iter', 'dali']


def build_data_loader(data_loader_type,
                      dataset,
                      batch_size,
                      repeat=1,
                      shuffle=True,
                      seed=0,
                      drop_last_sample=False,
                      drop_last_batch=True,
                      num_workers=0,
                      prefetch_factor=2,
                      pin_memory=False,
                      num_threads=1):
    """Builds a data loader with given dataset.

    Args:
        data_loader_type: Class type to which the data loader belongs, which is
            case insensitive.
        dataset: The dataset to load data from.
        batch_size: The batch size of the data produced by each replica.
        repeat: Repeating number of the entire dataset. (default: 1)
        shuffle: Whether to shuffle the samples within each epoch.
            (default: True)
        seed: Random seed used for shuffling. (default: 0)
        drop_last_sample: Whether to drop the tailing samples that cannot be
            evenly distributed. (default: False)
        drop_last_batch: Whether to drop the last incomplete batch.
            (default: True)
        num_workers: Number of workers to prefetch data for each replica.
            (default: 0)
        prefetch_factor: Number of samples loaded in advance by each worker.
            `N` means there will be a total of `N * num_workers` samples
            prefetched across all workers. (default: 2)
        pin_memory: Whether to use pinned memory for loaded data. This field is
            particularly used for `IterDataLoader`. (default: False)
        num_threads: Number of threads for each replica. This field is
            particularly used for `DALIDataLoader`. (default: 1)

    Raises:
        ValueError: If `data_loader_type` is not supported.
        NotImplementedError: If `data_loader_type` is not implemented yet.
    """
    data_loader_type = data_loader_type.lower()
    if data_loader_type not in _DATA_LOADERS_ALLOWED:
        raise ValueError(f'Invalid data loader type: `{data_loader_type}`!\n'
                         f'Types allowed: {_DATA_LOADERS_ALLOWED}.')

    if data_loader_type == 'dali' and DALIDataLoader is None:
        warnings.warn('DALI (Data Loading Library from NVIDIA) is not '
                      'supported on the current environment! '
                      'Fall back to `IterDataLoader`.')
        data_loader_type = 'iter'

    if data_loader_type == 'dali' and not dataset.support_dali:
        warnings.warn('DALI (Data Loading Library from NVIDIA) is not '
                      'supported by some transformation node of the dataset! '
                      'Fall back to `IterDataLoader`.')
        data_loader_type = 'iter'

    if data_loader_type == 'iter':
        return IterDataLoader(dataset=dataset,
                              batch_size=batch_size,
                              repeat=repeat,
                              shuffle=shuffle,
                              seed=seed,
                              drop_last_sample=drop_last_sample,
                              drop_last_batch=drop_last_batch,
                              num_workers=num_workers,
                              prefetch_factor=prefetch_factor,
                              pin_memory=pin_memory)
    if data_loader_type == 'dali':
        return DALIDataLoader(dataset=dataset,
                              batch_size=batch_size,
                              repeat=repeat,
                              shuffle=shuffle,
                              seed=seed,
                              drop_last_sample=drop_last_sample,
                              drop_last_batch=drop_last_batch,
                              num_workers=num_workers,
                              prefetch_factor=prefetch_factor,
                              num_threads=num_threads)
    raise NotImplementedError(f'Not implemented data loader type '
                              f'`{data_loader_type}`!')
