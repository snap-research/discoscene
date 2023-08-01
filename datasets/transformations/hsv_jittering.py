# python3.7
"""Implements image color jittering from the HSV space."""

import cv2
import numpy as np

try:
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
except ImportError:
    fn = None

from utils.formatting_utils import format_range
from .base_transformation import BaseTransformation

__all__ = ['HSVJittering']


class HSVJittering(BaseTransformation):
    """Applies random color jittering to images from the HSV space.

    Args:
        h_range: The range within which to uniformly sample a hue. Use `(0, 0)`
            to disable the hue jittering. (default: (0, 0))
        s_range: The range within which to uniformly sample a saturation. Use
            `(1, 1)` to disable the saturation jittering. (default: (1, 1))
        v_range: The range within which to uniformly sample a brightness value.
            Use `(1, 1)` to disable the brightness jittering. (default: (1, 1))
    """

    def __init__(self, h_range=(0, 0), s_range=(1, 1), v_range=(1, 1)):
        super().__init__(support_dali=(fn is not None))

        self.h_range = format_range(h_range)
        self.s_range = format_range(s_range, min_val=0)
        self.v_range = format_range(v_range, min_val=0)

    def _CPU_forward(self, data):
        # Early return if no jittering is needed.
        if (self.h_range == (0, 0) and self.s_range == (1, 1) and
                self.v_range == (1, 1)):
            return data

        # Get random jittering value for hue, saturation, and brightness.
        hue = np.random.uniform(*self.h_range)
        sat = np.random.uniform(*self.s_range)
        val = np.random.uniform(*self.v_range)

        # Perform color jittering.
        outputs = []
        for image in data:
            assert image.shape[2] == 3, 'RGB image is expected!'
            h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
            h = ((h + hue) % 180).astype(np.uint8)
            s = np.clip(s * sat, 0, 255).astype(np.uint8)
            v = np.clip(v * val, 0, 255).astype(np.uint8)
            new_image = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2RGB)
            outputs.append(new_image)
        return outputs

    def _DALI_forward(self, data):
        # Early return is no jittering is needed.
        if (self.h_range == (0, 0) and self.s_range == (1, 1) and
                self.v_range == (1, 1)):
            return data

        # Get random jittering value for hue, saturation, and brightness.
        if self.h_range[0] == self.h_range[1]:
            hue = self.h_range[0]
        else:
            hue = fn.random.uniform(range=self.h_range)
        hue = fn.cast(hue, dtype=types.FLOAT)
        if self.s_range[0] == self.s_range[1]:
            sat = self.s_range[0]
        else:
            sat = fn.random.uniform(range=self.s_range)
        sat = fn.cast(sat, dtype=types.FLOAT)
        if self.v_range[0] == self.v_range[1]:
            val = self.v_range[0]
        else:
            val = fn.random.uniform(range=self.v_range)
        val = fn.cast(val, dtype=types.FLOAT)

        # Perform color jittering.
        return fn.hsv(data, hue=hue, saturation=sat, value=val)
