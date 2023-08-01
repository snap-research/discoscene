# python3.7
"""Contains the base class for metric computation.

It will be of great importance to have a good metric to evaluate deep models.
Ideally, a good evaluation function is expected to give a convincing metric
given a pre-trained model and a validation dataset. To speed up the evaluation
process, distributed testing is supported, but NOT necessary.
"""

import os
import numpy as np

import torch
import torch.distributed as dist

from utils.misc import Infix

__all__ = ['BaseMetric']


class BaseMetric(object):
    """Defines the base metric class.

    Basically, each metric should implement the following functions:

    (1) evaluate(): Execute evaluation.
    (2) _is_better_than(): A helper function to compare two evaluation results.
    (3) save(): Save the evaluation results.
    (4) info(): Collect the information of the metric, like batch size and
        number of samples used.

    Meanwhile, each derived class should contain the following members:

    (1) name: Name of the metric, which is useful for printing log message.
    (2) work_dir: Working directory to save the results, and some temporary
        results. (default: None)
    (3) logger: A logger used to print log message and progressive bar.
        (default: None)
    (4) tb_writer: A TensorBoard writer used to record some results, which is
        optional. (default: None)
    (5) batch_size: The default batch size used for evaluation. This can be
        overwritten by the batch of `data_loader`. (default: 1)

    This base class also defines some helper functions to help dealing with
    distributed testing, including:

    (1) get_replica_num(): Get number of samples for the current replica.
    (2) get_indices(): Get sud-indices for the current replica.
    (3) pad_tensor(): Pad a tensor to ensure it has same shape across replicas.
    (4) gather_batch_results(): Gather the results across replicas per batch.
    (5) calibrate_index_order(): Calibrate the index order across replicas.
    (6) append_batch_results(): Append a batch results to the result list.
    (7) gather_all_results(): Gather the results from all batches.
    (8) sync(): Synchronize all replicas to make sure they are running into the
        same point.

    For example, the helper function can be used like:

    ```
    results = []
    for _ in range(len(val_data_loader)):
        batch_data = next(val_data_loader).cuda().detach()
        with torch.no_grad():
            batch_results = run_model(batch_data)
            # Padding can be skipped if all replicas are ensured to have same
            # number of samples.
            padded_results = self.pad_tensor(batch_results, self.batch_size)
            gathered_results = self.gather_batch_results(padded_results)
            self.append_batch_results(gathered_results, results)
    all_results = self.gather_all_results(results)
    if self.is_chief:
        compute_metric(all_results[:len(val_data_loader.dataset)])
    ```
    """

    def __init__(self,
                 name=None,
                 work_dir=None,
                 logger=None,
                 tb_writer=None,
                 batch_size=1):
        """Initializes the metric with basic settings."""
        self.name = name
        self.work_dir = work_dir
        self.logger = logger
        self.tb_writer = tb_writer
        self.batch_size = batch_size

        assert self.name is not None, 'Name is required!'
        assert self.work_dir is not None, 'Work directory is required!'
        assert self.logger is not None, 'Logger is required!'
        os.makedirs(self.work_dir, exist_ok=True)

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.device = torch.cuda.current_device()
        self.is_distributed = (self.world_size > 1)
        self.is_chief = (self.rank == 0)

        self.log_tail = f'for metric `{self.name}`'

    def get_replica_num(self, num):
        """Gets number of samples to process by the current replica.

        Args:
            num: The total number of samples to process by all replicas.

        Returns:
            Number of samples to process by each replica.
        """
        return (num + self.world_size - 1) // self.world_size

    def get_indices(self, num):
        """Gets sub-indices for the current replica.

        To make sure the samples are evenly distributed by all replicas, the
        indices may be padded with starting samples. For example, to distribute
        100 samples to 8 replicas, the indices will be

        [0, 1, 2, ..., 99, 0, 1, 2, 3]

        Args:
            num: The total number of samples to process by all replicas.

        Returns:
            A list of indices that should be processed by the current replica.
        """
        indices = list(range(num))
        if num % self.world_size > 0:
            total_num = (num // self.world_size + 1) * self.world_size
            indices += list(range(total_num - num))
        else:
            total_num = num
        return indices[self.rank:total_num:self.world_size]

    @staticmethod
    def pad_tensor(tensor, target_num):
        """Pads the first dimension of the input tensor to the target number.

        This function can be used to make sure the same tensor from different
        replicas have the same shape, which is essential for gathering.

        Args:
            tensor: The tensor to pad, if needed.
            target_num: Target first dimension of the padded tensor.
                Accordingly, this field is expected to be smaller than the first
                dimension of the given tensor.

        Returns:
            The padded tensor.
        """
        assert tensor.shape[0] <= target_num
        if tensor.shape[0] == target_num:
            return tensor

        pad_num = target_num - tensor.shape[0]
        pad_shape = list(tensor.shape)
        pad_shape[0] = pad_num
        padded_tensor = tensor.new(*pad_shape).fill_(0)
        return torch.cat((tensor, padded_tensor), dim=0)

    def gather_batch_results(self, results, calibrate_index=True):
        """Gathers the batch results from all replicas.

        Args:
            results: The tensor to gather, with shape [N, ...]
            calibrate_index: Whether to calibrate the index order after
                gathering. Please check function `self.calibrate_index_order()`
                for details. (default: True)

        Returns:
            A gathered results, with shape [N * num_replicas, ...], with type
                `numpy.ndarray`.
        """
        if not self.is_distributed:
            return results.detach().cpu().numpy()

        # Collect results across replicas.
        replica_results_list = []
        # NOTE: `torch.distributed.all_gather()` may only work for GPU data.
        # Hence we move the results onto GPU for gathering.
        results = results.cuda()
        for _ in range(self.world_size):
            replica_results_list.append(torch.zeros_like(results))
        dist.all_gather(replica_results_list, results.detach())
        all_results = torch.cat(replica_results_list, dim=0)

        if not self.is_chief:
            return None

        all_results = all_results.detach().cpu().numpy()
        # Calibrate the index order if needed.
        if calibrate_index:
            all_results = self.calibrate_index_order(all_results)

        return all_results

    def calibrate_index_order(self, batch_results_all_replicas):
        """Calibrates the index order across multiple replicas.

        Assuming there are two replicas in total, the first one deals with
        samples with index (0, 2, 4, 6, ...), while the other one deals with
        samples with index (1, 3, 5, 7, ...). But when gather the results, the
        index will be (0, 2, 4, 1, 3, 5, ...) if batch size equals to 3. This
        function is used to calibrate the index order back to (0, 1, 2, 3, ...).

        NOTE: This function should be called per batch.

        Args:
            batch_results_all_replicas: The results gathered from all replicas.
                It should be with shape [N * num_replicas, ...], where N is the
                batch size.

        Returns:
            Results with calibrated order, with shape [N * num_replicas, ...].
        """
        if not self.is_distributed:
            return batch_results_all_replicas

        indices = np.arange(batch_results_all_replicas.shape[0])
        indices = indices.reshape(self.world_size, -1)
        indices = indices.transpose(1, 0)
        indices = indices.flatten()
        return batch_results_all_replicas[indices]

    def append_batch_results(self, batch_results_all_replicas, result_list):
        """Appends batch results from all replicas to the result list.

        NOTE: This function is an in-place operation, where the new results will
        be appended to the result list.

        Args:
            batch_results_all_replicas: The results gathered from all replicas
                (order calibrated already if needed).
            result_list: A list to collect results from all batches. It should
                have already contained the results from previous batches.
        """
        if not self.is_chief:
            assert not batch_results_all_replicas  # Only chief owns results.
            assert not result_list  # Only chief processes the results.
            return
        result_list.append(batch_results_all_replicas)

    def gather_all_results(self, result_list):
        """Gathers the results from all batches.

        Args:
            result_list: A list of batch results.

        Returns:
            The gathered results.
        """
        if not self.is_chief:
            assert not result_list  # Only chief processes the results.
            return result_list
        return np.concatenate(result_list, axis=0)

    def sync(self):
        """Synchronizes all replicas."""
        if not self.is_distributed:
            return
        dist.barrier()

    def evaluate(self, *args):
        """Executes evaluation.

        NOTE: For numerical (quantitative) metrics, the returned result is
        recommended to be a dictionary, whose keys are metric names and values
        are evaluation results. This works best with `self._is_better_than()`
        for performance comparison.

        Args:
            *args: Should be like `data_loader, model_1, model_1_kwargs,
                model_2, model_2_kwargs, ...`.
        """
        raise NotImplementedError('Should be implemented in derived class!')

    def _is_better_than(self, metric_name, new, ref):
        """Determines the performance of two values given a particular metric.

        NOTE: Each derived metric class should override this function since
        different metrics can have different criteria to judge performance.

        Args:
            metric_name: Name of the metric used to judge the performance. This
                is of great use when a class groups a collection metrics.
            new: The new evaluation result.
            ref: The reference result as the baseline. If set as `None`, it
                means there is no reference performance yet, and the `new`
                should be set as the new baseline (i.e., return `True`).

        Returns:
            `True` if `new` is better than `ref`;
            `False` if `new` is worse than `ref`;
            `None` if `metric_name` does not support performance comparison.
        """
        raise NotImplementedError('Should be implemented in derived classes '
                                  'for evaluation results comparison!')

    def is_better_than(self, metric_name):
        """Wraps the function `_is_better_than()` with `Infix`.

        With the wrapping, it is possible to use

        ```
        flag = new << metric.is_better_than(metric_name) >> ref
        ```
        """
        fn = lambda new, ref: self._is_better_than(metric_name, new, ref)
        return Infix(fn)

    def save(self, result, target_filename=None, log_suffix=None, tag=None):
        """Saves the evaluation results from `self.evaluate()`.

        This function prints log message and also saves the result to the target
        file if needed.

        Args:
            result: The evaluation result outputted from `self.evaluate()`.
            target_filename: The name of the target file to save the result.
                The file will be save to `self.work_dir/${target_filename}.ext`,
                where the filename should not contain the extension. Instead,
                the extension is determined by each metric independently.
                (default: None)
            log_suffix: An optional suffix added to the log message.
                (default: None)
            tag: Tag (such as the running iteration), which is used to mark the
                evaluation. (default: None)
        """
        raise NotImplementedError('Should be implemented in derived class!')

    def info(self):
        """Collects the information of the metric.

        Please append new information in derived class if needed.
        """
        metric_info = {
            'Batch size': self.batch_size
        }
        return metric_info
