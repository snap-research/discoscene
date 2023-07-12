# python3.7
"""Contains the class of iteration-based data loader."""

from torch.utils.data import DataLoader

from .distributed_sampler import DistributedSampler
from .base_data_loader import BaseDataLoader

__all__ = ['IterDataLoader']


class IterDataLoader(BaseDataLoader):
    """Defines the iteration-based data loader."""

    def __init__(self,
                 dataset,
                 batch_size,
                 repeat=1,
                 shuffle=True,
                 seed=0,
                 drop_last_sample=False,
                 drop_last_batch=True,
                 num_workers=0,
                 prefetch_factor=2,
                 pin_memory=False):
        """Initializes the data loader.

        Args:
            pin_memory: Whether to use pinned memory for loaded data. If `True`,
                it will be faster to move data from CPU to GPU, however, it may
                require a high-performance computing system. (default: False)
        """
        self.pin_memory = pin_memory
        self._batch_grouper = None
        super().__init__(dataset=dataset,
                         batch_size=batch_size,
                         repeat=repeat,
                         shuffle=shuffle,
                         seed=seed,
                         drop_last_sample=drop_last_sample,
                         drop_last_batch=drop_last_batch,
                         num_workers=num_workers,
                         prefetch_factor=prefetch_factor)

    def __len__(self):
        return len(self._batch_grouper)

    def build(self):
        self._sampler = DistributedSampler(
            dataset=self._dataset,
            repeat=self.repeat,
            shuffle=self.shuffle,
            seed=self.seed,
            drop_last_sample=self.drop_last_sample,
            for_dali=False)
        self._batch_grouper = DataLoader(dataset=self._dataset,
                                         batch_size=self.batch_size,
                                         sampler=self._sampler,
                                         shuffle=False,
                                         drop_last=self.drop_last_batch,
                                         num_workers=self.num_workers,
                                         pin_memory=self.pin_memory,
                                         prefetch_factor=self.prefetch_factor)
        self._iter_loader = iter(self._batch_grouper)

    def reset_iter_loader(self):
        self._iter_loader = iter(self._batch_grouper)

    def info(self):
        data_loader_info = super().info()
        data_loader_info['Pin memory'] = self.pin_memory
        return data_loader_info
