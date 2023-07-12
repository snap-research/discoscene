# python3.7
"""Contains the distributed data sampler.

Basically, the sampler is responsible for maintaining a queue of indices, where
each index corresponds to a sample in the dataset. The samplers on all replicas
have exactly the same index order. Then, the first replica will fetch data
(0, R, 2R, ...), the second will fetch data (1, R + 1, 2R + 1, ...), given the
number of replicas, R.

However, sometimes, initialize the data loader and data sampler can be time
consuming (since it will load a large amount of data at one time). To avoid
re-initializing the data loader again and again, this sampler supports loading
the data for only one time and then repeating the indices. Please use the
argument `repeat` to control how many times data should be repeated. After
`repeat` times, the data will be re-loaded.

NOTE: The number of repeat times should NOT be very large, especially when there
are too many samples in the dataset. We recommend to set `repeat = 500` for
datasets with ~50K samples. If the dataset is mirrored, i.e., with
`dataset.mirror = True`, the repeat times should be halved accordingly.
"""

import torch
import torch.distributed as dist

__all__ = ['DistributedSampler']


class DistributedSampler(torch.utils.data.Sampler):
    """Implements the class for distributed sampling from dataset.

    This sampler will execute the following operations one by one:

    (1) Get the number of samples in the given dataset.
    (2) Get the initial list with `torch.arange()` for one epoch.
    (3) Repeat the list, and shuffle (if needed) for each repeated epoch.
    (4) Distribute the indices to multiple replicas, and drop the last if
        needed.

    NOTE: This class can also be used as the starting node of the data
    pre-processing graph in DALI (Data Loading Library from NVIDIA). Instead of
    using function `__iter__()` (as in `torch.utils.data.DataLoader`), DALI
    will use function `__call__()` to get a sample from the dataset. For more
    details, please refer to

    https://docs.nvidia.com/deeplearning/dali/user-guide/docs/

    Args:
        dataset: Dataset used for sampling.
        repeat: Repeating number of the entire dataset. (default: 1)
        shuffle: Whether to shuffle the indices within each epoch.
            (default: True)
        seed: Random seed used for shuffling. (default: 0)
        drop_last_sample: Whether to drop the tailing samples that cannot be
            evenly distributed. If set as `False`, the initial samples will be
            padded to the end to ensure the indices evenly divisible.
            (default: False)
        for_dali: Whether the sampler is for DALI or not. This field is
            particularly used to disable the function `__iter__()` to make DALI
            work properly. (default: False)
    """

    def __init__(self,
                 dataset,
                 repeat=1,
                 shuffle=True,
                 seed=0,
                 drop_last_sample=False,
                 for_dali=False):
        super().__init__(None)

        self._dataset = dataset
        self.repeat = max(1, int(repeat))
        self.shuffle = shuffle
        self.seed = max(0, seed)
        self.drop_last_sample = drop_last_sample
        self.for_dali = for_dali

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.dataset_length = len(self.dataset)
        self.actual_length = self.dataset_length * self.repeat

        if self.drop_last_sample:
            self.num_per_replica = self.actual_length // self.world_size
        else:
            self.num_per_replica = (
                (self.actual_length - 1) // self.world_size + 1)
        self.num_all_replicas = self.num_per_replica * self.world_size

        self.shuffle_times = 0  # How many times the dataset has been shuffled.
        self.indices = []
        if self.for_dali:
            self.generate_indices()

    def info(self):
        """Collects the information of the sampler."""
        sampler_info = {
            'Dataset repeat times': self.repeat,
            'Actual num samples': self.actual_length,
            'Shuffle': self.shuffle,
            'Seed': self.seed,
            'Drop unevenly distributed samples': self.drop_last_sample,
            'Num samples for all replicas': self.num_all_replicas,
            'Num sampler pre replica': self.num_per_replica
        }
        return sampler_info

    @property
    def dataset(self):
        """Returns the dataset."""
        return self._dataset

    def __len__(self):
        return self.num_per_replica

    def generate_indices(self):
        """Generates all the indices handled by the current replica."""
        indices = []
        g = torch.Generator()
        for _ in range(self.repeat):  # Repeat the dataset.
            if self.shuffle:
                g.manual_seed(self.seed + self.shuffle_times)
                self.shuffle_times += 1
                sub_indices = torch.randperm(self.dataset_length, generator=g)
            else:
                sub_indices = torch.arange(self.dataset_length)
            indices.extend(sub_indices.tolist())

        if self.drop_last_sample:
            indices = indices[:self.num_all_replicas]
        else:
            indices += indices[:(self.num_all_replicas - len(indices))]
        assert len(indices) == self.num_all_replicas

        # Sub-sampling for the current replica.
        self.indices = indices[self.rank:self.num_all_replicas:self.world_size]

    def __iter__(self):
        if self.for_dali:
            raise TypeError('Sampler for DALI is not iterable.')
        self.generate_indices()
        return iter(self.indices)

    def __call__(self, info):
        """This function is for fetching raw data for DALI pipeline.

        `info.idx_in_epoch` stands for the sample index for the current replica
        within the current epoch.
        """
        try:
            return self.dataset.get_raw_data(self.indices[info.idx_in_epoch])
        except IndexError:
            self.generate_indices()
            raise StopIteration  # pylint: disable=raise-missing-from
