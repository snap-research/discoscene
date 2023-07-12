# python3.7
"""Wraps a batch-based iterator introduced in DALI.

For more details, please refer to

https://docs.nvidia.com/deeplearning/dali/user-guide/docs/
"""

from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIGenericIterator

__all__ = ['DALIBatchIterator']


class DALIBatchIterator(DALIGenericIterator):
    """Defines the batch iterator for DALI data pre-processing.

    Args:
        pipeline: The pre-defined pipeline for data pre-processing.
        batch_size: Number of samples for each batch.
        drop_last_batch: Whether to drop the last incomplete batch.
            (default: True)
    """
    def __init__(self, pipeline, batch_size, drop_last_batch=True):
        self.batch_size = batch_size
        self.drop_last_batch = drop_last_batch

        if self.drop_last_batch:
            last_batch_padded = False
            last_batch_policy = LastBatchPolicy.FILL
            self.num_batches = len(pipeline) // batch_size
        else:
            last_batch_padded = True
            last_batch_policy = LastBatchPolicy.DROP
            self.num_batches = (len(pipeline) - 1) // batch_size + 1

        super().__init__(pipelines=pipeline,
                         size=-1,
                         auto_reset=False,
                         output_map=pipeline.dataset.output_keys,
                         last_batch_padded=last_batch_padded,
                         last_batch_policy=last_batch_policy,
                         prepare_first_batch=True)

    def __next__(self):
        # [0] means the first GPU. In the distributed case, each replica only
        # has one GPU.
        return super().__next__()[0]

    def __len__(self):
        return self.num_batches
