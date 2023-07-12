# python3.7
"""Contains the base class for model conversion.

Basically, all converters aim at converting pre-trained models (usually
officially released or trained with officially released code) to match the
networks defined in `../models/`. After that, the converted weights can be
easily loaded and used for other purpose.

To make sure the conversion is correct, each converter should also maintain a
function to compare the results before and after the conversion when using the
same input for inference (forward), and a function to compare the gradients
before and after the conversion when using the same input, loss function, and
optimizer for back-propagation (backward).
"""

import numpy as np
import torch

from models import build_model
from utils.loggers import build_logger

__all__ = ['BaseConverter']


class BaseConverter(object):
    """Defines the base converter for converting model weights.

    A converter should have the following members:

    (1) verbose_log: Whether to print verbose log messages. (default: False)

    A converter should have the following functions:

    (1) load_source(): Loads source weights.
    (2) convert(): Converts the weights.
    (3) save_target(): Saves target weights.
    (4) test_forward(): Check the conversion in the forward process.
    (5) test_backward(): Check the conversion in the backward process.
    (6) run(): The main running function.

    NOTE: All tests will only yield the comparison error, but will NOT make the
    judgement on whether the conversion is correct. Users should make the call
    by themselves.
    """

    def __init__(self, verbose_log=False):
        """Initializes the converter with a logger."""
        self.logger = build_logger(logfile=None, verbose_log=verbose_log)

        # The pre-trained weights may contain more than one model.
        self.src_path = None
        self.dst_path = None
        self.src_models = dict()
        self.dst_models = dict()
        self.dst_kwargs = dict()
        self.dst_states = dict()

        # Whether the source model is executable? This field is required by
        # conversion test.
        self.source_executable = True

        # To make sure the reproducibility.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    @staticmethod
    def to_numpy(data):
        """Converts the input data to `numpy.ndarray`."""
        if isinstance(data, (int, float)):
            return np.array(data)
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        raise TypeError(f'Not supported data type `{type(data)}` for '
                        f'converting to `numpy.ndarray`!')

    def mean_error(self, data_1, data_2=None):
        """Computes the mean absolute error between two data."""
        data_1 = self.to_numpy(data_1).astype(np.float64)
        if data_2 is None:
            return np.mean(np.abs(data_1))
        data_2 = self.to_numpy(data_2).astype(np.float64)
        assert data_1.shape == data_2.shape
        return np.mean(np.abs(data_1 - data_2))

    def max_error(self, data_1, data_2=None):
        """Computes the maximum absolute error between two data."""
        data_1 = self.to_numpy(data_1).astype(np.float64)
        if data_2 is None:
            return np.max(np.abs(data_1))
        data_2 = self.to_numpy(data_2).astype(np.float64)
        assert data_1.shape == data_2.shape
        return np.max(np.abs(data_1 - data_2))

    def check_weight(self, src_state, dst_state, init_state=None):
        """Checks whether all weight parameters from two given states are close.

        This function can be used to compare the gradients in the
        back-propagation process as long as the source model and the target
        model are using the same optimizer, where the native Gradient Descent is
        recommended.

        NOTE: This function will also compare the source state to the initial
        state to trace the life-ong weight change if `init_state` is provided.

        Args:
            src_state: The source weights, in the dictionary format.
            dst_state: The target weights, in the dictionary format.
            init_state: The initial state. If set as `None`, the comparison to
                the initial state will be skipped. (default: None)
        """
        trace_init = (init_state is not None)

        assert len(src_state) == len(src_state)
        for key, src in src_state.items():
            dst = dst_state[key]
            init = init_state[key] if trace_init else None
            if not torch.allclose(src, dst, equal_nan=True):
                mean_error = self.mean_error(src, dst)
                max_error = self.max_error(src, dst)
                log_message = (
                    f'Error (mean: {mean_error:.3e}, max: {max_error:.3e})')
                if trace_init:
                    mean_change = self.mean_error(src, init)
                    max_change = self.max_error(src, init)
                    log_message += (f',    Life-long change ('
                                    f'mean: {mean_change:.3e}, '
                                    f'max: {max_change:.3e})')
                src_mean = self.mean_error(src, None)
                log_message += (f',    Source mean: {src_mean:.3e}'
                                f'     from parameter `{key}`.')
                self.logger.print(log_message, indent_level=3)

    def load_source(self, path):
        """Loads source weights from the given path.

        Args:
            path: Path to the pre-trained weights to load from.
        """
        raise NotImplementedError('Should be implemented in derived classes!')

    def parse_model_config(self):
        """Parses configurations from the source model.

        This is particular used for building target model.
        """
        raise NotImplementedError('Should be implemented in derived classes!')

    def build_target(self):
        """Builds the target model with parsed configuration."""
        for model_name in self.src_models:
            self.dst_models[model_name] = build_model(
                **self.dst_kwargs[model_name])

    def convert(self):
        """Converts the model weights."""
        raise NotImplementedError('Should be implemented in derived classes!')

    def save_target(self, path):
        """Saves the converted weights to the given path.

        Args:
            path: Path to save the converted weights.
        """
        save_state = {
            'models': self.dst_states,
            'model_kwargs_init': self.dst_kwargs
        }
        torch.save(save_state, path)

    def test_forward(self, num, save_test_image=False):
        """Tests the conversion results with forward.

        NOTE: Batch size is set to 1 by default.

        Args:
            num: Number of samples (iterations) used for forward test.
            save_test_image: Whether to save the intermediate image results if
                needed. This is particularly used for generative models.
                (default: False)
        """
        raise NotImplementedError('Should be implemented in derived classes!')

    def test_backward(self, num, learning_rate=0.01):
        """Tests the conversion results with backward.

        NOTE: Batch size is set to 1 by default.

        Args:
            num: Number of samples (iterations) used for backward test.
            learning_rate: Learning rate used for back-propagation.
                (default: 0.01)
        """
        raise NotImplementedError('Should be implemented in derived classes!')

    def run(self,
            src_path,
            dst_path,
            forward_test_num=0,
            backward_test_num=0,
            save_test_image=False,
            learning_rate=0.01):
        """The main running function."""
        self.src_path = src_path
        self.dst_path = dst_path

        self.logger.print('========================================')
        self.logger.print(f'Dealing with `{src_path}` ...')

        self.logger.print('--------------------', is_verbose=True)
        self.logger.print(f'Loading weights from `{src_path}` ...',
                          is_verbose=True)
        self.load_source(src_path)
        self.logger.print('Successfully loaded!', is_verbose=True)

        self.logger.print('--------------------', is_verbose=True)
        self.logger.print('Converting weights ...', is_verbose=True)
        self.convert()
        self.logger.print('Successfully Converted!', is_verbose=True)

        self.logger.print('--------------------', is_verbose=True)
        self.logger.print(f'Saving weights to `{dst_path}` ...',
                          is_verbose=True)
        self.save_target(dst_path)
        self.logger.print('Successfully saved!', is_verbose=True)

        if forward_test_num > 0 and self.source_executable:
            self.logger.print('--------------------')
            self.logger.print('Testing conversion (forward) ...')
            self.test_forward(forward_test_num, save_test_image=save_test_image)

        if backward_test_num > 0 and self.source_executable:
            self.logger.print('--------------------')
            self.logger.print('Testing conversion (backward) ...')
            self.test_backward(backward_test_num, learning_rate=learning_rate)

        self.logger.print('========================================')
