# python3.7
"""Implements image normalization."""

import numpy as np

try:
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
except ImportError:
    fn = None

from .base_transformation import BaseTransformation

__all__ = ['Normalize']


class Normalize(BaseTransformation):
    """Normalizes images.

    The input images is expected to with pixel range [0, 255].

    The output images will be with data format `CHW` and dtype `float32`.

    Args:
        min_val: The minimum value after normalization. (default: -1.0)
        max_val: The maximum value after normalization. (default: 1.0)
    """

    def __init__(self, min_val=-1.0, max_val=1.0):
        super().__init__(support_dali=(fn is not None))

        self.min_val = float(min_val)
        self.max_val = float(max_val)

    def _CPU_forward(self, data):
        outputs = []
        for image in data:
            image = image.astype(np.float32)
            image = image / 255 * (self.max_val - self.min_val) + self.min_val
            image = image.transpose(2, 0, 1)
            outputs.append(image)
        return outputs

    def _DALI_forward(self, data):
        return fn.crop_mirror_normalize(
            data,
            dtype=types.FLOAT,
            output_layout=types.NCHW,
            scale=(self.max_val - self.min_val) / 255.0,
            shift=self.min_val)
