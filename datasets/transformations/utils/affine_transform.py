# python3.7
"""Contains the function to generate a random affine transformation."""

import cv2
import numpy as np

from utils.formatting_utils import format_range
from utils.formatting_utils import format_image_size

__all__ = ['generate_affine_transformation']


def generate_affine_transformation(image_size,
                                   rotation_range,
                                   scale_range,
                                   tx_range,
                                   ty_range):
    """Generates a random affine transformation matrix.

    Args:
        image_size: Size of the image, which is used as a reference of the
            transformation. The size is assumed with order (height, width).
        rotation_range: The range (in degrees) within which to uniformly sample
            a rotation angle.
        scale_range: The range within which to uniformly sample a scaling
            factor.
        tx_range: The range (in length of image width) within which to uniformly
            sample a X translation.
        ty_range: The range (in length of image height) within which to
            uniformly sample a Y translation.

    Returns:
        A transformation matrix, with shape [2, 3] and dtype `numpy.float32`.
    """
    # Regularize inputs.
    height, width = format_image_size(image_size)
    rotation_range = format_range(rotation_range)
    scale_range = format_range(scale_range, min_val=0)
    tx_range = format_range(tx_range)
    ty_range = format_range(ty_range)

    # Sample parameters for the affine transformation.
    rotation = np.random.uniform(*rotation_range)
    scale = np.random.uniform(*scale_range)
    tx = np.random.uniform(*tx_range)
    ty = np.random.uniform(*ty_range)

    # Get the transformation matrix.
    matrix = cv2.getRotationMatrix2D(center=(width // 2, height // 2),
                                     angle=rotation,
                                     scale=scale)
    matrix[:, 2] += (tx * width, ty * height)

    return matrix.astype(np.float32)
