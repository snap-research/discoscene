# python3.7
"""Collects all transformations for data pre-processing."""

from .affine_transform import AffineTransform
from .blur_and_sharpen import BlurAndSharpen
from .crop import CenterCrop
from .crop import RandomCrop
from .crop import LongSideCrop
from .decode import Decode
from .flip import Flip
from .hsv_jittering import HSVJittering
from .jpeg_compress import JpegCompress
from .normalize import Normalize
from .region_brightness import RegionBrightness
from .resize import Resize
from .resize import ProgressiveResize
from .resize import ResizeAug
from .resize import ResizeShortside

try:
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
except ImportError:
    fn = None

__all__ = ['build_transformation', 'switch_between']


_TRANSFORMATIONS = {
    'AffineTransform': AffineTransform,
    'BlurAndSharpen': BlurAndSharpen,
    'CenterCrop': CenterCrop,
    'RandomCrop': RandomCrop,
    'LongSideCrop': LongSideCrop,
    'Decode': Decode,
    'Flip': Flip,
    'HSVJittering': HSVJittering,
    'JpegCompress': JpegCompress,
    'Normalize': Normalize,
    'RegionBrightness': RegionBrightness,
    'Resize': Resize,
    'ProgressiveResize': ProgressiveResize,
    'ResizeAug': ResizeAug,
    'ResizeShortside': ResizeShortside
}


def build_transformation(transform_type, **kwargs):
    """Builds a transformation based on its class type.

    Args:
        transform_type: Class type to which the transformation belongs,
            which is case sensitive.
        **kwargs: Additional arguments to build the transformation.

    Raises:
        ValueError: If the `transform_type` is not supported.
    """
    if transform_type not in _TRANSFORMATIONS:
        raise ValueError(f'Invalid transformation type: '
                         f'`{transform_type}`!\n'
                         f'Types allowed: {list(_TRANSFORMATIONS)}.')
    return _TRANSFORMATIONS[transform_type](**kwargs)


def switch_between(cond, cond_true, cond_false, use_dali=False):
    """Switches between two transformation nodes for data pre-processing.

    Args:
        cond: Condition to switch between two alternatives.
        cond_true: The returned value if the condition fulfills.
        cond_false: The returned value if the condition fails.
        use_dali: Whether the nodes are from DALI pre-processing pipeline.
            (default: False)

    Returns:
        One of `cond_true` and `cond_false`, depending on `cond`.
    """
    if use_dali and fn is None:
        raise NotImplementedError('DALI is not supported! '
                                  'Please install first.')

    if not use_dali:
        return cond_true if cond else cond_false

    cond = fn.cast(cond, dtype=types.BOOL)
    return cond_true * cond + cond_false * (cond ^ True)
