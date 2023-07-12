# python3.7
"""Contains utility functions used for formatting."""

import cv2
import numpy as np

__all__ = [
    'format_time', 'format_range', 'format_image_size', 'format_image',
    'raw_label_to_one_hot', 'one_hot_to_raw_label'
]


def format_time(seconds):
    """Formats seconds to readable time string.

    Args:
        seconds: Number of seconds to format.

    Returns:
        The formatted time string.

    Raises:
        ValueError: If the input `seconds` is less than 0.
    """
    if seconds < 0:
        raise ValueError(f'Input `seconds` should be greater than or equal to '
                         f'0, but `{seconds}` is received!')

    # Returns seconds as float if less than 1 minute.
    if seconds < 10:
        return f'{seconds:7.3f} s'
    if seconds < 60:
        return f'{seconds:7.2f} s'

    seconds = int(seconds + 0.5)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f'{days:2d} d {hours:02d} h'
    if hours:
        return f'{hours:2d} h {minutes:02d} m'
    return f'{minutes:2d} m {seconds:02d} s'


def format_range(obj, min_val=None, max_val=None):
    """Formats the given object to a valid range.

    If `min_val` or `max_val` is provided, both the starting value and the end
    value will be clamped to range `[min_val, max_val]`.

    NOTE: (a, b) is regarded as a valid range if and only if `a <= b`.

    Args:
        obj: The input object to format.
        min_val: The minimum value to cut off the input range. If not provided,
            the default minimum value is negative infinity. (default: None)
        max_val: The maximum value to cut off the input range. If not provided,
            the default maximum value is infinity. (default: None)

    Returns:
        A two-elements tuple, indicating the start and the end of the range.

    Raises:
        ValueError: If the input object is an invalid range.
    """
    if not isinstance(obj, (tuple, list)):
        raise ValueError(f'Input object must be a tuple or a list, '
                         f'but `{type(obj)}` received!')
    if len(obj) != 2:
        raise ValueError(f'Input object is expected to contain two elements, '
                         f'but `{len(obj)}` received!')
    if obj[0] > obj[1]:
        raise ValueError(f'The second element is expected to be equal to or '
                         f'greater than the first one, '
                         f'but `({obj[0]}, {obj[1]})` received!')

    obj = list(obj)
    if min_val is not None:
        obj[0] = max(obj[0], min_val)
        obj[1] = max(obj[1], min_val)
    if max_val is not None:
        obj[0] = min(obj[0], max_val)
        obj[1] = min(obj[1], max_val)
    return tuple(obj)


def format_image_size(size):
    """Formats the given image size to a two-element tuple.

    A valid image size can be an integer, indicating both the height and the
    width, OR can be a two-element list or tuple. Both height and width are
    assumed to be positive integer.

    Args:
        size: The input size to format.

    Returns:
        A two-elements tuple, indicating the height and the width, respectively.

    Raises:
        ValueError: If the input size is invalid.
    """
    if not isinstance(size, (int, tuple, list)):
        raise ValueError(f'Input size must be an integer, a tuple, or a list, '
                         f'but `{type(size)}` received!')
    if isinstance(size, int):
        size = (size, size)
    else:
        if len(size) == 1:
            size = (size[0], size[0])
        if not len(size) == 2:
            raise ValueError(f'Input size is expected to have two numbers at '
                             f'most, but `{len(size)}` numbers received!')
    if not isinstance(size[0], int) or size[0] < 0:
        raise ValueError(f'The height is expected to be a non-negative '
                         f'integer, but `{size[0]}` received!')
    if not isinstance(size[1], int) or size[1] < 0:
        raise ValueError(f'The width is expected to be a non-negative '
                         f'integer, but `{size[1]}` received!')
    return tuple(size)


def format_image(image):
    """Formats an image read from `cv2`.

    NOTE: This function will always return a 3-dimensional image (i.e., with
    shape [H, W, C]) in pixel range [0, 255]. For color images, the channel
    order of the input is expected to be with `BGR` or `BGRA`, which is the
    raw image decoded by `cv2`; while the channel order of the output is set to
    `RGB` or `RGBA` by default.

    Args:
        image: `np.ndarray`, an image read by `cv2.imread()` or
            `cv2.imdecode()`.

    Returns:
        An image with shape [H, W, C] (where `C = 1` for grayscale image).
    """
    if image.ndim == 2:  # add additional axis if given a grayscale image
        image = image[:, :, np.newaxis]

    assert isinstance(image, np.ndarray)
    assert image.dtype == np.uint8
    assert image.ndim == 3 and image.shape[2] in [1, 3, 4]

    if image.shape[2] == 3:  # BGR image
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[2] == 4:  # BGRA image
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return image


def raw_label_to_one_hot(raw_label, num_classes):
    """Converts a single label into one-hot vector.

    Args:
        raw_label: The raw label.
        num_classes: Total number of classes.

    Returns:
        one-hot vector of the given raw label.
    """
    if len(raw_label) == 1:
        one_hot = np.zeros(num_classes, dtype=np.float32)
        one_hot[raw_label] = 1.0
    elif len(raw_label)>1:
        one_hot = np.zeros((len(raw_label), num_classes), dtype=np.float32)
        arange_idx = np.arange(len(raw_label))
        try:
            raw_label = raw_label.astype(np.int32)
            one_hot[arange_idx, raw_label] = 1.0
        except:
            import ipdb;ipdb.set_trace()

    return one_hot


def one_hot_to_raw_label(one_hot):
    """Converts a one-hot vector to a single value label.

    Args:
        one_hot: `np.ndarray`, a one-hot encoded vector.

    Returns:
        A single integer to represent the category.
    """
    return np.argmax(one_hot)
