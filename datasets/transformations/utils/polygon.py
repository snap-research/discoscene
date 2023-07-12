# python3.7
"""Contains the functions to generate a random polygon area in an image."""

import cv2
import numpy as np

from utils.formatting_utils import format_range
from utils.formatting_utils import format_image_size

__all__ = ['generate_polygon_contour', 'generate_polygon_mask']


def generate_polygon_contour(center_x,
                             center_y,
                             num_vertices,
                             avg_radius,
                             spikyness,
                             irregularity):
    """Generates a random polygon contour, represented by a set of vertices.

    Starting with the center `(center_x, center_y)`, this function creates a
    random polygon by sampling points on a circle around the center.

    In particular, `num_vertices` (N) points will be sampled in sequence. For
    each point, its distance to the center (i.e., radius), and its rotation
    angle referring to its previous point, are sampled independently.

    More concretely,

    - the radius is sampled subject to a normal distribution, with mean
        `avg_radius` and variance `spikyness`, and then clipped to range
        `[0, 2 * avg_radius]`.
    - the angle is sampled subject to a uniform distribution, with range
        `[2 * pi / N - irregularity, 2 * pi / N + irregularity]`.

    To make sure that the sampled N points exactly form a circle (i.e., 2 * pi),
    the sampled angles will be summed together and then normalized. Also, the
    initial point with be placed at an angle uniformly sampled from [0, 2 * pi].

    Args:
        center_x: X coordinate of the polygon center.
        center_y: Y coordinate of the polygon center.
        num_vertices: Number of vertices used to generate the polygon.
        avg_radius: The average radius (in pixels) of the created polygon, which
            controls the mean distance between each vertex to the center.
        spikyness: Variance of the radius of each sampled vertex. This field
            will be first clipped to range [0, 1], and then mapped to
            [0, `avg_radius`].
        irregularity: Variance of the rotation angle of each sampled vertex.
            This field will be first clipped to range [0, 1], and then mapped to
            [0, 2 * pi / `num_vertices`]

    Returns:
        An array with shape [N, 2], representing the (x, y) coordinates of
            vertices in counterclockwise order, and with dtype `numpy.int64`.
    """
    # Regularize inputs.
    center_x = float(center_x)
    center_y = float(center_y)
    num_vertices = int(num_vertices)
    avg_radius = float(avg_radius)
    spikyness = float(spikyness)
    irregularity = float(irregularity)
    assert num_vertices > 2, 'At least three points for a polygon!'
    assert avg_radius > 0, 'Average radius should be positive!'

    # Sample the radius for each vertex.
    spikyness = np.clip(spikyness, 0, 1) * avg_radius
    radii = avg_radius + np.random.normal(size=num_vertices) * spikyness
    radii = np.clip(radii, 0, 2 * avg_radius)

    # Sample the rotation angle for each vertex.
    avg_rotation = 2 * np.pi / num_vertices
    irregularity = np.clip(irregularity, 0, 1) * avg_rotation
    randomness = np.random.uniform(size=num_vertices)
    rotations = avg_rotation + randomness * irregularity * 2 - irregularity
    rotations = rotations / np.sum(rotations) * 2 * np.pi  # normalize

    # Sample the starting angle of the initial vertex.
    init_angle = np.random.uniform(0, 2 * np.pi)
    angles = np.cumsum(rotations) + init_angle

    # Compute the coordinates of each vertex.
    coordinates = np.zeros(shape=(num_vertices, 2), dtype=np.float64)
    coordinates[:, 0] = center_x + radii * np.cos(angles)
    coordinates[:, 1] = center_y + radii * np.sin(angles)

    return (coordinates + 0.5).astype(np.int64)


def generate_polygon_mask(image_size,
                          image_channels,
                          center_x_range,
                          center_y_range,
                          num_vertices,
                          radius_range,
                          spikyness_range,
                          irregularity_range,
                          max_blur_kernel_ratio,
                          min_blur_kernel_size=3,
                          blur_x_std=3,
                          blur_y_std=None):
    """Generates a random polygon mask with target image size.

    This function first randomly generates a polygon contour (i.e., a set of
    vertices), then fills the polygon area with 1 and remaining area with 0, and
    finally filters the mask with a Gaussian kernel if needed. More concretely,
    this function calls the following helper functions in sequence:

    1. generate_polygon_contour()
    2. cv2.fillPoly()
    3. cv2.GaussianBlur()

    Args:
        image_size: Size of the image, which is used as a reference of the
            generated mask. The size is assumed with order (height, width).
        image_channels: Number of image channels. If this field is specified as
            a positive number, the output mask will be repeated along the
            channel dimension. If set as `None` or a non-positive number, the
            output mask will be with shape [height, width].
        center_x_range: The range within which to uniformly sample an X position
            of the polygon center. This field takes image width as the unit
            length.
        center_y_range: The range within which to uniformly sample a Y position
            of the polygon center. This field takes image height as the unit
            length.
        num_vertices: Number of vertices used to generate the polygon. See
            function `generate_polygon_contour()` for more details.
        radius_range: The range within which to uniformly sample an average
            radius of th polygon. This field takes the short side of the image
            as the unit length. See function `generate_polygon_contour()` for
            more details.
        spikyness_range: The range within which to uniformly sample a variance
            of the radius of each sampled vertex. See function
            `generate_polygon_contour()` for more details.
        irregularity_range: The range within which to uniformly sample a
            variance of the rotation angle of each sampled vertex. See function
            `generate_polygon_contour()` for more details.
        max_blur_kernel_ratio: Ratio the control the maximum size of the
            blurring kernel. Set this field as `0` to skip filtering.
        min_blur_kernel_size: The minimum size of the blurring kernel.
            (default: 3)
        blur_x_std: The standard deviation of blurring kernel in X direction.
            (default: 3)
        blur_y_std: The standard deviation of blurring kernel in Y direction.
            If not specified, `blur_x_std` will be used. (default: None)

    Returns:
        A mask with shape [H, W] if `image_channels` is invalid, or with shape
            [H, W, C] is `image_channels` is valid, and with dtype
            `numpy.float32`.
    """
    # Regularize inputs.
    height, width = format_image_size(image_size)
    min_size = min(height, width)
    max_size = max(height, width)
    if image_channels is None:
        image_channels = 0
    image_channels = int(image_channels)
    if image_channels > 0:
        assert image_channels in [1, 3, 4], 'Support Gray, RGB, RGBA images!'
    center_x_range = format_range(center_x_range, min_val=0, max_val=1)
    center_y_range = format_range(center_y_range, min_val=0, max_val=1)
    radius_range = format_range(radius_range, min_val=0, max_val=1)
    spikyness_range = format_range(spikyness_range, min_val=0, max_val=1)
    irregularity_range = format_range(irregularity_range, min_val=0, max_val=1)
    max_blur_kernel_ratio = float(max_blur_kernel_ratio)
    min_blur_kernel_size = max(0, int(min_blur_kernel_size))

    # Sample hyper-parameters for the polygon region.
    center_x = int(np.random.uniform(*center_x_range) * width + 0.5)
    center_y = int(np.random.uniform(*center_y_range) * height + 0.5)
    radius = int(np.random.uniform(*radius_range) * min_size + 0.5)
    spikyness = np.random.uniform(*spikyness_range)
    irregularity = np.random.uniform(*irregularity_range)
    max_blur_kernel_size = int(np.ceil(max_size * max_blur_kernel_ratio))
    if max_blur_kernel_size > min_blur_kernel_size:
        ksize = np.random.randint(min_blur_kernel_size, max_blur_kernel_size)
    elif max_blur_kernel_size == min_blur_kernel_size:
        ksize = min_blur_kernel_size
    else:
        ksize = 0

    # Generate polygon mask.
    vertices = generate_polygon_contour(center_x=center_x,
                                        center_y=center_y,
                                        num_vertices=num_vertices,
                                        avg_radius=radius,
                                        spikyness=spikyness,
                                        irregularity=irregularity)
    mask = np.zeros((height, width), dtype=np.float64)
    cv2.fillPoly(mask, pts=[vertices], color=(1.0))
    if ksize > 0:
        mask = cv2.GaussianBlur(mask,
                                ksize=(ksize * 2 + 1, ksize * 2 + 1),
                                sigmaX=blur_x_std,
                                sigmaY=blur_y_std)

    if image_channels > 0:
        mask = np.repeat(mask[:, :, None], image_channels, axis=2)
    return mask.astype(np.float32)
