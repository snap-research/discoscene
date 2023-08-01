# python3.7
"""Contains the base class for data transformation.

When training deep models, data transformation, which is also known as data
pre-processing or data augmentation, is critical. Users can choose to execute
the transformation on CPU (with `numpy` and `cv2`) or GPU (with DALI). With
either option, the key design is to form a transformation pipeline, where each
node is one of unit transformations. This file provides a base class for these
unit transformations.

For more details on using DALI, please refer to

https://docs.nvidia.com/deeplearning/dali/user-guide/docs/
"""

try:
    import nvidia.dali.pipeline as dali_pipeline
except ImportError:
    dali_pipeline = None

__all__ = ['BaseTransformation']


class BaseTransformation(object):
    """Defines the base class for data transformation.

    Data can include images, labels, etc. For simplicity, images are used as the
    example to explain the function of this class.

    Basically, to transform an image (or a list of images), this class provides
    a `__call__()` function to pre-process raw data. Here, please note that each
    derived transformation class is supposed to process ONE sample (instead of
    a batch of samples) at one time. But, as mentioned above, the class can also
    handle a list of images. That is because, one data entry (i.e., sample) can
    consist of multiple images, like the image pair for pix2pix training.
    Randomness may be introduced in the transformation process, e.g., random
    augmentation. Hence, if users want to process all images within one data
    entry with exactly the same manner, it is possible to first group them into
    a list and call the function `__call__()` ONLY once. Otherwise, calling the
    function `__call__()` for each image separately may cause different
    augmentations to them.

    To transform an image (or a list of images), two types of forwarding are
    supported, i.e., the typical CPU function with `numpy` and `cv2`, and the
    faster GPU function with DALI. Each derived transformation class is
    supposed to support CPU forward, i.e., `self._CPU_forward()`, and is
    optional to support DALI forward, i.e., `self._DALI_forward()`. Please make
    sure these two types of forwarding have identical behavior. Also, a derived
    class is assumed to NOT support DALI by default. Please use
    `support_dali = True` to initialize the class if `self._DALI_forward()` is
    implemented.

    To implements DALI pre-processing, please refer to

    https://docs.nvidia.com/deeplearning/dali/user-guide/docs/

    NOTE: The transformation class also maintains a member, called
    `self._has_customized_function_for_dali`. This field affects how the DALI
    pipeline is setup. Basically, if some transformation uses customized python
    function for forwarding, i.e., `nvidia.dali.fn.python_function`, the
    pipeline should turn off `exec_async` and `exec_pipelined`. This field is
    set to `False` by default, and MUST be overridden as `True` if customized
    function is used for `self._DALI_forward()`. Unfortunately, when this field
    is set as `True`, parallel data pre-fetching may also be turned off. Please
    refer to `datasets/data_loaders/dali_pipeline.py` for more details.
    """

    def __init__(self, support_dali=False):
        """Initializes the class by indicating whether DALI is supported."""
        self._name = self.__class__.__name__
        self._support_dali = support_dali
        self._has_customized_function_for_dali = False

    @property
    def name(self):
        """Returns the name of the transformation."""
        return self._name

    @property
    def support_dali(self):
        """Whether the transformation supports DALI forwarding."""
        return self._support_dali

    @property
    def has_customized_function_for_dali(self):
        """Whether DALI forwarding is implemented with customized function."""
        return self._has_customized_function_for_dali

    def _CPU_forward(self, data):
        """Transforms the input data with typical CPU operations.

        NOTE: It is strongly recommended to use `numpy` and `cv2` to process
        images.

        Args:
            data: A list of `numpy.ndarray`.
        """
        raise NotImplementedError('Should be implemented in derived class!')

    def _DALI_forward(self, data):
        """Transforms the input data with DALI operations.

        NOTE: DALI forward is based on a pre-compiled static graph. Hence, all
        operations should be considered as a node in the graph, instead of
        run-time operations.

        Args:
            data: A list of `nvidia.dali.pipeline.DataNode`.
        """
        raise NotImplementedError(f'DALI forward is not supported in '
                                  f'data transformation `{self.name}`!')

    def __call__(self, data, use_dali=False):
        """Transforms the input data with the proper manner.

        Basically, this function chooses between `self._CPU_forward()` and
        `self._DALI_forward()`. In addition, this function handles the case
        where `data` is not a list, such that `self._CPU_forward()` and
        `self._DALI_forward()` only need to consider list input.

        Args:
            use_dali: Whether to use `self._DALI_forward()` for forwarding or
                not. (default: False)
        """
        is_input_list = True

        if not isinstance(data, (list, tuple)):
            is_input_list = False
            data = [data]

        if use_dali and self.support_dali and dali_pipeline is not None:
            outputs = self._DALI_forward(data)
        else:
            outputs = self._CPU_forward(data)

        if not is_input_list and isinstance(outputs, (list, tuple)):
            return outputs[0]
        return outputs
