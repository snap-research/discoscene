# python3.7
"""Collects all datasets."""

from .image_dataset import ImageDataset
from .image_bbox_dataset import ImageBboxDataset
from .data_loaders import build_data_loader

__all__ = ['build_dataset']

_DATASETS = {
    'ImageDataset': ImageDataset,
    'ImageBboxDataset': ImageBboxDataset,

}


def build_dataset(for_training,
                  batch_size,
                  dataset_kwargs=None,
                  data_loader_kwargs=None,
                  dataset_only=False):
    """Builds a dataset with a data loader.

    Args:
        for_training: Whether the dataset is used for training or not.
        batch_size: Bach size of the built data loader.
        dataset_kwargs: A dictionary, containing the arguments for building
            dataset.
        data_loader_kwargs: A dictionary, containing the arguments for building
            data loader. (default: None)

    Returns:
        A built data loader.

    Raises:
        ValueError: If the input `batch_size` is invalid.
        ValueError: If `dataset_kwargs` is not provided, or it does not have the
            key `dataset_type`, or the provided `dataset_type` is not supported.
        ValueError: If `data_loader_kwargs` is not provided, or it does not have
            the key `data_loader_type`.
    """
    for_training = bool(for_training)

    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f'Batch size should be a positive integer, '
                         f'but `{batch_size}` is received!')

    if not isinstance(dataset_kwargs, dict):
        raise ValueError(f'`dataset_kwargs` should be a dictionary, '
                         f'but `{type(dataset_kwargs)}` is received!')
    if 'dataset_type' not in dataset_kwargs:
        raise ValueError('`dataset_type` is not found in `dataset_kwargs`!')

    if not dataset_only:
        if not isinstance(data_loader_kwargs, dict):
            raise ValueError(f'`data_loader_kwargs` should be a dictionary, '
                             f'but `{type(data_loader_kwargs)}` is received!')
        if 'data_loader_type' not in data_loader_kwargs:
            raise ValueError('`data_loader_type` is not found in '
                             '`data_loader_kwargs`!')

    # Build dataset.
    dataset_type = dataset_kwargs.pop('dataset_type')
    if dataset_type not in _DATASETS:
        raise ValueError(f'Invalid dataset type: `{dataset_type}`!\n'
                         f'Types allowed: {list(_DATASETS)}.')
    dataset = _DATASETS[dataset_type](**dataset_kwargs)
    if dataset_only:
        return dataset
    # Build data loader based on the dataset.
    data_loader_type = data_loader_kwargs.pop('data_loader_type')
    if for_training:
        data_loader_kwargs['shuffle'] = True
        data_loader_kwargs['drop_last_sample'] = False
        data_loader_kwargs['drop_last_batch'] = True
    else:
        data_loader_kwargs['repeat'] = 1
        data_loader_kwargs['shuffle'] = False
        data_loader_kwargs['drop_last_sample'] = False
        data_loader_kwargs['drop_last_batch'] = False

    return build_data_loader(data_loader_type=data_loader_type,
                             dataset=dataset,
                             batch_size=batch_size,
                             **data_loader_kwargs)
