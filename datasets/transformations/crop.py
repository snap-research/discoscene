# python3.7
"""Implements image cropping."""

import numpy as np

try:
    import nvidia.dali.fn as fn
    import cupy
except ImportError:
    fn = None

from utils.formatting_utils import format_image_size
from .base_transformation import BaseTransformation

__all__ = ['CenterCrop', 'RandomCrop', 'LongSideCrop']


class CenterCrop(BaseTransformation):
    """Applies central cropping to images.

    Args:
        crop_size: Size of the cropped image, which is assumed with order
            (height, width).
    """

    def __init__(self, crop_size):
        super().__init__(support_dali=(fn is not None))

        self.crop_size = format_image_size(crop_size)

    def _CPU_forward(self, data):
        outputs = []
        for image in data:
            height, width = image.shape[:2]
            if height == self.crop_size[0] and width == self.crop_size[1]:
                outputs.append(image)
                continue
            if height < self.crop_size[0]:
                raise ValueError(f'Cropping height `{self.crop_size[0]}` is '
                                 f'larger than image height `{height}`!')
            if width < self.crop_size[1]:
                raise ValueError(f'Cropping width `{self.crop_size[1]}` is '
                                 f'larger than image width `{width}`!')
            y = (height - self.crop_size[0]) // 2
            x = (width - self.crop_size[1]) // 2
            outputs.append(np.ascontiguousarray(
                image[y:y + self.crop_size[0], x:x + self.crop_size[1]]))
        return outputs

    def _DALI_forward(self, data):
        return fn.crop(data,
                       crop_pos_x=0.5,
                       crop_pos_y=0.5,
                       crop_w=self.crop_size[1],
                       crop_h=self.crop_size[0],
                       out_of_bounds_policy='error')


class RandomCrop(BaseTransformation):
    """Applies random cropping to images.

    Args:
        crop_size: Size of the cropped image, which is assumed with order
            (height, width).
    """

    def __init__(self, crop_size):
        super().__init__(support_dali=(fn is not None))

        self.crop_size = format_image_size(crop_size)

    def _CPU_forward(self, data):
        crop_pos_y = np.random.uniform()
        crop_pos_x = np.random.uniform()

        outputs = []
        for image in data:
            height, width = image.shape[:2]
            if height == self.crop_size[0] and width == self.crop_size[1]:
                outputs.append(image)
                continue
            if height < self.crop_size[0]:
                raise ValueError(f'Cropping height `{self.crop_size[0]}` is '
                                 f'larger than image height `{height}`!')
            if width < self.crop_size[1]:
                raise ValueError(f'Cropping width `{self.crop_size[1]}` is '
                                 f'larger than image width `{width}`!')
            y = int((height - self.crop_size[0]) * crop_pos_y)
            x = int((width - self.crop_size[1]) * crop_pos_x)
            outputs.append(np.ascontiguousarray(
                image[y:y + self.crop_size[0], x:x + self.crop_size[1]]))
        return outputs

    def _DALI_forward(self, data):
        crop_pos_y = fn.random.uniform(range=(0, 1))
        crop_pos_x = fn.random.uniform(range=(0, 1))
        return fn.crop(data,
                       crop_pos_x=crop_pos_x,
                       crop_pos_y=crop_pos_y,
                       crop_w=self.crop_size[1],
                       crop_h=self.crop_size[0],
                       out_of_bounds_policy='error')


class LongSideCrop(BaseTransformation):
    """Crops a square region from images along the long side.

    The length of the short side will be kept.

    NOTE: This transformation applies a customized python operation (with CuPy)
    for DALI forwarding, which may disable parallel data pre-processing.

    Args:
        center_crop: Whether to centrally crop the image along the long side.
            (default: True)
    """

    def __init__(self, center_crop=True):
        super().__init__(support_dali=(fn is not None))
        self._has_customized_function_for_dali = True

        self.center_crop = center_crop

    def _CPU_forward(self, data):
        if self.center_crop:
            crop_pos = 0.5
        else:
            crop_pos = np.random.uniform()

        outputs = []
        for image in data:
            height, width = image.shape[:2]
            if height == width:
                outputs.append(image)
                continue
            crop_size = min(height, width)
            y = int((height - crop_size) * crop_pos)
            x = int((width - crop_size) * crop_pos)
            outputs.append(np.ascontiguousarray(
                image[y:y + crop_size, x:x + crop_size]))
        return outputs

    def _DALI_forward(self, data):
        # Defines a helper function implemented with cupy.
        def helper(*images):
            if self.center_crop:
                crop_pos = 0.5
            else:
                crop_pos = cupy.random.uniform()

            outputs = []
            for image in images:
                height, width = image.shape[:2]
                if height == width:
                    outputs.append(image)
                    continue
                crop_size = min(height, width)
                y = int((height - crop_size) * crop_pos)
                x = int((width - crop_size) * crop_pos)
                outputs.append(cupy.ascontiguousarray(
                    image[y:y + crop_size, x:x + crop_size]))
            return tuple(outputs)

        return fn.python_function(
            *data, device='gpu', function=helper, num_outputs=len(data))
