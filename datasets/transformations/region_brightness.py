# python3.7
"""Implements brightness adjustment of a certain image region."""

import numpy as np

try:
    import nvidia.dali.fn as fn
    import nvidia.dali.math as dmath
    import nvidia.dali.types as types
except ImportError:
    fn = None

from utils.formatting_utils import format_range
from utils.formatting_utils import format_image_size
from .utils import generate_polygon_mask
from .base_transformation import BaseTransformation

__all__ = ['RegionBrightness']


class RegionBrightness(BaseTransformation):
    """Applies random brightness adjustment to particular image region.

    The image region is randomly sampled as a polygon. Please refer to
    `datasets/transformations/utils/polygon.py` for more details.

    Args:
        image_size: Size of the image, which is used as a reference of the
            generated mask. The size is assumed with order (height, width).
        image_channels: Number of image channels. If this field is specified as
            a positive number, the output mask will be repeated along the
            channel dimension. If set as `None` or a non-positive number, the
            output mask will be with shape [height, width]. (default: 3)
        prob: Probability of applying brightness adjustment. (default: 0.3)
        brightness_change: The maximum allowed brightness change. (default 0.6)
        center_x_range: The range within which to uniformly sample an X position
            of the polygon center. This field takes image width as the unit
            length. (default: (0.2, 0.8))
        center_y_range: The range within which to uniformly sample a Y position
            of the polygon center. This field takes image height as the unit
            length. (default: (0.25, 0.75))
        num_vertices: Number of vertices used to generate the polygon.
            (default: 40)
        radius_range: The range within which to uniformly sample an average
            radius of th polygon. This field takes the short side of the image
            as the unit length. (default: (0, 0.25))
        spikyness_range: The range within which to uniformly sample a variance
            of the radius of each sampled vertex. default: (0.1, 0.1))
        irregularity_range: The range within which to uniformly sample a
            variance of the rotation angle of each sampled vertex.
            (default: (0, 1))
        max_blur_kernel_ratio: Ratio the control the maximum size of the
            blurring kernel. Set this field as `0` to skip filtering.
            (default: 0.025)
        min_blur_kernel_size: The minimum size of the blurring kernel.
            (default: 3)
        blur_x_std: The standard deviation of blurring kernel in X direction.
            (default: 3)
        blur_y_std: The standard deviation of blurring kernel in Y direction.
            If not specified, `blur_x_std` will be used. (default: None)
        prefetch_queue_depth: Depth of the prefetch queue. (default: 32)
    """

    def __init__(self,
                 image_size,
                 image_channels=3,
                 prob=0.3,
                 brightness_change=0.6,
                 center_x_range=(0.2, 0.8),
                 center_y_range=(0.25, 0.75),
                 num_vertices=40,
                 radius_range=(0, 0.25),
                 spikyness_range=(0.1, 0.1),
                 irregularity_range=(0, 1),
                 max_blur_kernel_ratio=0.025,
                 min_blur_kernel_size=3,
                 blur_x_std=3,
                 blur_y_std=None,
                 prefetch_queue_depth=32):
        super().__init__(support_dali=(fn is not None))

        self.image_size = format_image_size(image_size)
        self.image_channels = image_channels
        self.prob = np.clip(prob, 0, 1)
        self.brightness_change = brightness_change
        self.center_x_range = format_range(center_x_range, min_val=0, max_val=1)
        self.center_y_range = format_range(center_y_range, min_val=0, max_val=1)
        self.num_vertices = num_vertices
        self.radius_range = format_range(radius_range, min_val=0, max_val=1)
        self.spikyness_range = format_range(
            spikyness_range, min_val=0, max_val=1)
        self.irregularity_range = format_range(
            irregularity_range, min_val=0, max_val=1)
        self.max_blur_kernel_ratio = max_blur_kernel_ratio
        self.min_blur_kernel_size = min_blur_kernel_size
        self.blur_x_std = blur_x_std
        self.blur_y_std = blur_y_std
        self.prefetch_queue_depth = prefetch_queue_depth

        # The lambda function is particularly used to get compatible with DALI.
        self.generate_polygon_fn = lambda _input: generate_polygon_mask(
            image_size=self.image_size,
            image_channels=self.image_channels,
            center_x_range=self.center_x_range,
            center_y_range=self.center_y_range,
            num_vertices=self.num_vertices,
            radius_range=self.radius_range,
            spikyness_range=self.spikyness_range,
            irregularity_range=self.irregularity_range,
            max_blur_kernel_ratio=self.max_blur_kernel_ratio,
            min_blur_kernel_size=self.min_blur_kernel_size,
            blur_x_std=self.blur_x_std,
            blur_y_std=self.blur_y_std)

    def _CPU_forward(self, data):
        # Early return if no brightness adjustment is applied.
        if np.random.uniform() >= self.prob:
            return data

        # Prepare polygon region and brightness adjustment strength.
        mask = self.generate_polygon_fn(None)
        mask = mask * np.random.uniform(-1, 1) * self.brightness_change

        # Adjust image brightness within a certain region.
        outputs = []
        for image in data:
            image = image.astype(np.float32)
            image = np.clip(image - image * mask, 0, 255)
            image = image.astype(np.uint8)
            outputs.append(image)
        return outputs

    def _DALI_forward(self, data):
        # Prepare polygon region and brightness adjustment strength.
        mask = fn.external_source(
            source=self.generate_polygon_fn,
            parallel=True,
            prefetch_queue_depth=self.prefetch_queue_depth,
            batch=False)
        strength = fn.random.uniform(range=(-1, 1))
        mask = mask * strength * self.brightness_change

        # Adjust image brightness within a certain region.
        cond = fn.random.coin_flip(dtype=types.BOOL, probability=self.prob)
        outputs = []
        for image in data:
            adjusted_image = fn.cast(image, dtype=types.FLOAT)
            adjusted_image = dmath.clamp(
                adjusted_image - adjusted_image * mask, 0, 255)
            adjusted_image = fn.cast(adjusted_image, dtype=types.UINT8)
            outputs.append(adjusted_image * cond + image * (cond ^ True))
        return outputs
