# python3.7
"""Contains the class of DALI-based data loader.

For more details, please refer to

https://docs.nvidia.com/deeplearning/dali/user-guide/docs/
"""

from .dali_batch_iterator import DALIBatchIterator
from .dali_pipeline import DALIPipeline
from .distributed_sampler import DistributedSampler
from .base_data_loader import BaseDataLoader

__all__ = ['DALIDataLoader']


class DALIDataLoader(BaseDataLoader):
    """Defines the DALI-based data loader."""

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
                 num_threads=1):
        """Initializes the data loader.

        Args:
            num_threads: Number of threads used for each replica. (default: 1)
        """
        self.num_threads = num_threads
        self._pipeline = None
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
        return len(self.iter_loader)

    def build(self):
        self._sampler = DistributedSampler(
            dataset=self._dataset,
            shuffle=self.shuffle,
            repeat=self.repeat,
            seed=self.seed,
            drop_last_sample=self.drop_last_sample,
            for_dali=True)
        prefetch_queue_depth = max(1, self.num_workers * self.prefetch_factor)
        self._pipeline = DALIPipeline(dataset=self._dataset,
                                      sampler=self._sampler,
                                      batch_size=self.batch_size,
                                      seed=self.seed,
                                      num_workers=self.num_workers,
                                      num_threads=self.num_threads,
                                      prefetch_queue_depth=prefetch_queue_depth)
        self._iter_loader = DALIBatchIterator(
            pipeline=self._pipeline,
            batch_size=self.batch_size,
            drop_last_batch=self.drop_last_batch)

    def reset_iter_loader(self):
        self._iter_loader.reset()

    def info(self):
        data_loader_info = super().info()
        data_loader_info['Num threads'] = self.num_threads
        return data_loader_info
