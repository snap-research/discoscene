# python3.7
"""Implements image resizing."""

import cv2
import numpy as np

try:
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
except ImportError:
    fn = None

from utils.formatting_utils import format_range
from utils.formatting_utils import format_image_size
from .base_transformation import BaseTransformation

__all__ = ['Resize', 'ResizeShortside', 'ProgressiveResize', 'ResizeAug']


class Resize(BaseTransformation):
    """Applies image resizing.

    Args:
        image_size: Size of the output image.
    """

    def __init__(self, image_size):
        super().__init__(support_dali=(fn is not None))
        self.image_size = format_image_size(image_size)

    def _CPU_forward(self, data):
        outputs = []
        for image in data:
            if image.shape[:2] == self.image_size:
                outputs.append(image)
                continue
            outputs.append(
                cv2.resize(image, (self.image_size[1], self.image_size[0]),
                           interpolation=cv2.INTER_AREA))
        return outputs

    def _DALI_forward(self, data):
        return fn.resize(data,
                         resize_y=float(self.image_size[0]),
                         resize_x=float(self.image_size[1]),
                         interp_type=types.INTERP_LANCZOS3)

class ResizeShortside(BaseTransformation):
    """Applies image resizing.

    Args:
        image_size: Size of the output image.
    """

    def __init__(self, image_size):
        super().__init__(support_dali=(fn is not None))
        self.image_size = format_image_size(image_size)

    def _CPU_forward(self, data):
        outputs = []
        for image in data:
            if image.shape[:2] == self.image_size:
                outputs.append(image)
                continue
            h, w = image.shape[:2]
            short_side = self.image_size[0]
            if w < h:
                width = short_side
                height = int(short_side * h / w)
            else:
                height = short_side
                width = int(short_side * w / h)
            outputs.append(
                cv2.resize(image, (width, height),
                           interpolation=cv2.INTER_AREA))
        return outputs

class ProgressiveResize(BaseTransformation):
    """Applies image resizing progressively.

    Different from normal resize, this transformation will reduce the image size
    progressively. In each step, the maximum reduce factor is 2.

    NOTE: This class can only handle square images currently, and can only be
    used for downsampling.

    NOTE: DALI is not supported by this class.

    Args:
        image_size: Size of the output image.
    """

    def __init__(self, image_size):
        super().__init__(support_dali=False)
        self.image_size = format_image_size(image_size)

        if not self.image_size[0] == self.image_size[1]:
            raise ValueError(f'Only square size is supported, but '
                             f'height ({self.image_size[0]}) and '
                             f'width ({self.image_size[1]}) are received!')

    def _CPU_forward(self, data):
        size = self.image_size[0]

        outputs = []
        for image in data:
            height, width = image.shape[:2]
            assert height == width, 'Only support square image!'
            assert height >= size, 'Only support downsampling!'
            while height > size:
                height = max(height // 2, size)
                image = cv2.resize(image, (height, height),
                                   interpolation=cv2.INTER_LINEAR)
            outputs.append(image)
        return outputs

    def _DALI_forward(self, data):
        raise NotImplementedError(f'DALI forward is not supported in '
                                  f'data transformation `{self.name}`!')


class ResizeAug(BaseTransformation):
    """Applies resize augmentation to images.

    This augmentation will randomly downsample the image and then resize it
    back to the input size.

    Args:
        image_size: Size of the input and the output image, which is assumed
            with order (height, width).
        prob: Probability of applying the augmentation. (default: 0.5)
        down_range: The range within which to uniformly sample a downsampling
            factor. (default: (1, 2.5))
    """

    def __init__(self, image_size, prob=0.5, down_range=(1, 2.5)):
        super().__init__(support_dali=(fn is not None))

        self.image_size = format_image_size(image_size)
        self.prob = np.clip(prob, 0, 1)
        self.down_range = format_range(down_range, min_val=1)

    def _CPU_forward(self, data):
        # Sample a downsampling factor.
        ratio = np.random.uniform(*self.down_range)
        down_height = int(self.image_size[0] / ratio + 0.5)
        down_width = int(self.image_size[1] / ratio + 0.5)

        outputs = []
        for image in data:
            image = cv2.resize(image, (down_width, down_height),
                               interpolation=cv2.INTER_AREA)
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]),
                               interpolation=cv2.INTER_NEAREST)
            outputs.append(image)

        return outputs

    def _DALI_forward(self, data):
        # Sample a downsampling factor.
        if self.down_range[0] == self.down_range[1]:
            ratio = self.down_range[0]
        else:
            ratio = fn.random.uniform(range=self.down_range)
        ratio = fn.cast(ratio, dtype=types.FLOAT)
        down_height = fn.cast(self.image_size[0] / ratio, dtype=types.FLOAT)
        down_width = fn.cast(self.image_size[1] / ratio, dtype=types.FLOAT)
        down_data = fn.resize(data,
                              resize_y=down_height,
                              resize_x=down_width,
                              interp_type=types.INTERP_LANCZOS3)
        up_data = fn.resize(down_data,
                            resize_y=float(self.image_size[0]),
                            resize_x=float(self.image_size[1]),
                            interp_type=types.INTERP_NN)

        # Determine whether the augmentation should be applied.
        cond = fn.random.coin_flip(dtype=types.BOOL, probability=self.prob)
        return up_data * cond + data * (cond ^ True)
