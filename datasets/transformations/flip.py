# python3.7
"""Implements image flipping."""

import numpy as np

try:
    import nvidia.dali.fn as fn
except ImportError:
    fn = None

from .base_transformation import BaseTransformation

__all__ = ['Flip']


class Flip(BaseTransformation):
    """Applies random flipping to images.

    Args:
        horizontal_prob: Probability of flipping images horizontally.
            (default: 0.0)
        vertical_prob: Probability of flipping images vertically. (default: 0.0)
    """

    def __init__(self, horizontal_prob=0.0, vertical_prob=0.0):
        super().__init__(support_dali=(fn is not None))

        self.horizontal_prob = np.clip(horizontal_prob, 0, 1)
        self.vertical_prob = np.clip(vertical_prob, 0, 1)

    def _CPU_forward(self, data):
        do_horizontal = np.random.uniform() < self.horizontal_prob
        do_vertical = np.random.uniform() < self.vertical_prob

        # Early return if no flipping is applied.
        if not do_horizontal and not do_vertical:
            return data

        outputs = []
        for image in data:
            if do_horizontal:
                image = image[:, ::-1]
            if do_vertical:
                image = image[::-1, :]
            outputs.append(np.ascontiguousarray(image))
        return outputs

    def _DALI_forward(self, data):
        do_horizontal = fn.random.coin_flip(probability=self.horizontal_prob)
        do_vertical = fn.random.coin_flip(probability=self.vertical_prob)
        return fn.flip(data, horizontal=do_horizontal, vertical=do_vertical)
