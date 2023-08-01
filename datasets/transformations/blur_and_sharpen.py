# python3.7
"""Implements image blurring and sharpening."""

import cv2
import numpy as np

try:
    import nvidia.dali.fn as fn
    import nvidia.dali.math as dmath
    import nvidia.dali.types as types
except ImportError:
    fn = None

from utils.formatting_utils import format_range
from .base_transformation import BaseTransformation

__all__ = ['BlurAndSharpen']


class BlurAndSharpen(BaseTransformation):
    """Applies random blurring and sharpening to images.

    NOTE: Image sharpening is only applied if no blurring happens. In other
    words, an image will NEVER be first blurred then sharpened.

    Args:
        blur_prob: Probability of applying image blurring. (default: 0.5)
        sharpen_prob: Probability of applying image sharpening. (default: 0.5)
        kernel_range: The range (in pixels) within which to uniformly sample a
            kernel size. This kernel size is used for both image blurring and
            image sharpening. (default (3, 7))
        sharpen_range: The range within which to uniformly sample a sharpen
            strength. (default: (1.5, 2.0))
    """

    def __init__(self,
                 blur_prob=0.5,
                 sharpen_prob=0.5,
                 kernel_range=(3, 7),
                 sharpen_range=(1.5, 2.0)):
        super().__init__(support_dali=(fn is not None))

        self.blur_prob = np.clip(blur_prob, 0, 1)
        self.sharpen_prob = np.clip(sharpen_prob, 0, 1)
        self.kernel_range = format_range(kernel_range, min_val=0)
        self.sharpen_range = format_range(sharpen_range, min_val=0)

    def _CPU_forward(self, data):
        do_blur = np.random.uniform() < self.blur_prob
        do_sharpen = (np.random.uniform() < self.sharpen_prob) and not do_blur

        # Early return if neither blurring nor sharpening is applied.
        if not do_blur and not do_sharpen:
            return data

        # Get settings for blurring and sharpening.
        ksize = int(np.random.uniform(*self.kernel_range)) * 2 + 1
        sharpen_strength = np.random.uniform(*self.sharpen_range)

        # Execute blurring or sharpening.
        outputs = []
        for image in data:
            if do_blur:  # Blurring is applied.
                image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=0)
            else:  # Sharpening is applied.
                blurred_image = cv2.GaussianBlur(
                    image, (ksize, ksize), sigmaX=0)
                image = image.astype(np.float32)
                image = image + (image - blurred_image) * sharpen_strength
                image = np.clip(image + 0.5, 0, 255).astype(np.uint8)
            outputs.append(image)
        return outputs

    def _DALI_forward(self, data):
        blur_cond = fn.random.coin_flip(dtype=types.BOOL,
                                        probability=self.blur_prob)
        sharpen_cond = fn.random.coin_flip(dtype=types.BOOL,
                                           probability=self.sharpen_prob)
        sharpen_cond = sharpen_cond & (blur_cond ^ True)

        # Get settings for blurring and sharpening.
        if self.kernel_range[0] == self.kernel_range[1]:
            ksize = self.kernel_range[0]
        else:
            ksize = fn.random.uniform(range=self.kernel_range)
        ksize = fn.cast(ksize, dtype=types.INT32) * 2 + 1
        if self.sharpen_range[0] == self.sharpen_range[1]:
            sharpen_strength = self.sharpen_range[0]
        else:
            sharpen_strength = fn.random.uniform(range=self.sharpen_range)
        sharpen_strength = fn.cast(sharpen_strength, dtype=types.FLOAT)

        outputs = []
        for image in data:
            # Random blurring.
            blurred_image = fn.gaussian_blur(image, window_size=ksize)
            image = blurred_image * blur_cond + image * (blur_cond ^ True)
            # Random sharpening.
            temp_image = fn.cast(
                fn.gaussian_blur(image, window_size=ksize), dtype=types.FLOAT)
            float_image = fn.cast(image, dtype=types.FLOAT)
            sharpened_image = fn.cast(
                dmath.clamp(
                    float_image + (float_image - temp_image) * sharpen_strength,
                    0,
                    255),
                dtype=types.UINT8)
            outputs.append(
                sharpened_image * sharpen_cond + image * (sharpen_cond ^ True))
        return outputs
