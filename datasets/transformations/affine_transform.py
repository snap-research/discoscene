# python3.7
"""Implements image affine transformation."""

import cv2
import numpy as np

try:
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
except ImportError:
    fn = None

from utils.formatting_utils import format_range
from utils.formatting_utils import format_image_size
from .utils import generate_affine_transformation
from .base_transformation import BaseTransformation

__all__ = ['AffineTransform']


class AffineTransform(BaseTransformation):
    """Applies a random affine transformation to images.

    Args:
        image_size: Size of the image, which is used as a reference of the
            transformation. The size is assumed with order (height, width).
        prob: Probability of applying affine transformation. (default: 0.5)
        rotation_range: The range (in degrees) within which to uniformly sample
            a rotation angle. (default: (-15, 15))
        scale_range: The range within which to uniformly sample a scaling
            factor. (default: (0.95, 1.05))
        tx_range: The range (in length of image width) within which to uniformly
            sample a X translation. (default: (-0.02, 0.02))
        ty_range: The range (in length of image height) within which to
            uniformly sample a Y translation. (default: (-0.02, 0.02))
        prefetch_queue_depth: Depth of the prefetch queue. (default: 32)
    """

    def __init__(self,
                 image_size,
                 prob=0.5,
                 rotation_range=(-15, 15),
                 scale_range=(0.95, 1.05),
                 tx_range=(-0.02, 0.02),
                 ty_range=(-0.02, 0.02),
                 prefetch_queue_depth=32):
        super().__init__(support_dali=(fn is not None))

        self.image_size = format_image_size(image_size)
        self.prob = np.clip(prob, 0, 1)
        self.rotation_range = format_range(rotation_range)
        self.scale_range = format_range(scale_range, min_val=0)
        self.tx_range = format_range(tx_range)
        self.ty_range = format_range(ty_range)
        self.prefetch_queue_depth = prefetch_queue_depth

        # The lambda function is particularly used to get compatible with DALI.
        self.generate_affine_fn = lambda _input: generate_affine_transformation(
            image_size=self.image_size,
            rotation_range=self.rotation_range,
            scale_range=self.scale_range,
            tx_range=self.tx_range,
            ty_range=self.ty_range)

    def _CPU_forward(self, data):
        # Early return if no affine transformation is needed.
        if np.random.uniform() >= self.prob:
            return data

        # Prepare random affine transformation matrix.
        transformation_matrix = self.generate_affine_fn(None)
        height, width = self.image_size

        # Warp images.
        outputs = []
        for image in data:
            outputs.append(cv2.warpAffine(image,
                                          transformation_matrix,
                                          dsize=(width, height),
                                          flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=0))
        return outputs

    def _DALI_forward(self, data):
        # Prepare random affine transformation matrix.
        transformation_matrix = fn.external_source(
            source=self.generate_affine_fn,
            parallel=True,
            prefetch_queue_depth=self.prefetch_queue_depth,
            batch=False)

        # Warp images.
        transformed_data = fn.warp_affine(data,
                                          transformation_matrix.gpu(),
                                          size=self.image_size,
                                          interp_type=types.INTERP_LINEAR,
                                          fill_value=0,
                                          inverse_map=False)

        # Determine whether the transformation should be applied.
        cond = fn.random.coin_flip(dtype=types.BOOL, probability=self.prob)
        return transformed_data * cond + data * (cond ^ True)
