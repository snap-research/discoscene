# python3.7
"""Wraps the data pre-processing pipeline introduced in DALI.

DALI (Data Loading Library from NVIDIA) deploys the data pre-processing
pipeline on GPU instead of CPU for acceleration. It relies on a pre-compiling
graph. This file wraps this pipeline to fit the data loader.

For more details, please refer to

https://docs.nvidia.com/deeplearning/dali/user-guide/docs/
"""

import dill

import nvidia.dali.ops as ops
from nvidia.dali.pipeline import Pipeline

__all__ = ['DALIPipeline']


class DALIPipeline(Pipeline):
    """Defines the pipeline for DALI data pre-processing.

    Args:
        dataset: Dataset to load data from.
        sampler: Index sampler.
        batch_size: Number of samples for each batch.
        seed: Seed for randomness in data pre-processing. (default: 0)
        num_workers: Number of workers to pre-fetch data (on CPU) from the
            dataset. (default: 0)
        num_threads: Number of threads used for data pre-processing by the
            current replica. (default: 1)
        prefetch_queue_depth: Prefetch queue depth. (default: 1)
    """

    def __init__(self,
                 dataset,
                 sampler,
                 batch_size,
                 seed=0,
                 num_workers=0,
                 num_threads=1,
                 prefetch_queue_depth=1):
        self._dataset = dataset
        self._sampler = sampler

        # Starting node of the data pre-processing graph.
        self.get_raw_data = ops.ExternalSource(
            source=self.sampler,
            num_outputs=self.dataset.num_raw_outputs,
            parallel=True,
            prefetch_queue_depth=prefetch_queue_depth)

        if seed >= 0:
            seed = seed * self.sampler.world_size + self.sampler.rank
        else:
            seed = -1

        if dataset.has_customized_function_for_dali:
            exec_pipelined = False
            exec_async = False
            num_workers = 0
        else:
            exec_pipelined = True
            exec_async = True
        super().__init__(batch_size=batch_size,
                         num_threads=num_threads,
                         device_id=self.sampler.rank,
                         seed=seed,
                         exec_pipelined=exec_pipelined,
                         exec_async=exec_async,
                         py_num_workers=num_workers,
                         py_start_method='spawn',
                         py_callback_pickler=dill)

    @property
    def dataset(self):
        """Returns the dataset."""
        return self._dataset

    @property
    def sampler(self):
        """Returns the sampler."""
        return self._sampler

    def define_graph(self):
        return self.dataset.define_dali_graph(self.get_raw_data())

    def __len__(self):
        return len(self.sampler)
