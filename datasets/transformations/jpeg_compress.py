# python3.7
"""Implements JPEG compression on images."""

import cv2
import numpy as np

try:
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
except ImportError:
    fn = None

from utils.formatting_utils import format_range
from utils.formatting_utils import format_image
from .base_transformation import BaseTransformation

__all__ = ['JpegCompress']


class JpegCompress(BaseTransformation):
    """Applies random JPEG compression to images.

    This transformation can be used as an augmentation by distorting images.
    In other words, the input image(s) will be first compressed (i.e., encoded)
    with a random quality ratio, and then decoded back to the image space.
    The distortion is introduced in the encoding process.

    Args:
        quality_range: The range within which to uniformly sample a quality
            value after compression. 100 means highest and 0 means lowest.
            (default: (40, 60))
        prob: Probability of applying JPEG compression. (default: 0.5)
    """

    def __init__(self, prob=0.5, quality_range=(40, 60)):
        super().__init__(support_dali=(fn is not None))

        self.prob = np.clip(prob, 0, 1)
        self.quality_range = format_range(quality_range, min_val=0, max_val=100)

    def _CPU_forward(self, data):
        # Early return if no compression is needed.
        if np.random.uniform() >= self.prob:
            return data

        # Set compression quality.
        quality = np.random.randint(*self.quality_range)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

        # Compress images.
        outputs = []
        for image in data:
            _, encoded_image = cv2.imencode('.jpg', image, encode_param)
            decoded_image = format_image(
                cv2.imdecode(encoded_image, cv2.IMREAD_UNCHANGED))
            outputs.append(decoded_image)
        return outputs

    def _DALI_forward(self, data):
        # Set compression quality.
        if self.quality_range[0] == self.quality_range[1]:
            quality = self.quality_range[0]
        else:
            quality = fn.random.uniform(range=self.quality_range)
        quality = fn.cast(quality, dtype=types.INT32)

        # Compress images.
        compressed_images = fn.jpeg_compression_distortion(
            data, quality=quality)

        # Determine whether the transformation should be applied.
        cond = fn.random.coin_flip(dtype=types.BOOL, probability=self.prob)
        return compressed_images * cond + data * (cond ^ True)
