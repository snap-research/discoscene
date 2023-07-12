# python3.7
"""Contains the visualizer to visualize images as a GIF."""

from PIL import Image

from ..image_utils import parse_image_size
from ..image_utils import load_image
from ..image_utils import resize_image
from ..image_utils import list_images_from_dir

__all__ = ['GifVisualizer']


class GifVisualizer(object):
    """Defines the visualizer that visualizes an image collection as GIF."""

    def __init__(self, image_size=None, duration=100, loop=0):
        """Initializes the GIF visualizer.

        Args:
            image_size: Size for image visualization. (default: None)
            duration: Duration between two frames, in milliseconds.
                (default: 100)
            loop: How many times to loop the GIF. `0` means infinite.
                (default: 0)
        """
        self.set_image_size(image_size)
        self.set_duration(duration)
        self.set_loop(loop)

    def set_image_size(self, image_size=None):
        """Sets the image size of the GIF."""
        height, width = parse_image_size(image_size)
        self.image_height = height
        self.image_width = width

    def set_duration(self, duration=100):
        """Sets the GIF duration."""
        self.duration = duration

    def set_loop(self, loop=0):
        """Sets how many times the GIF will be looped. `0` means infinite."""
        self.loop = loop

    def visualize_collection(self, images, save_path):
        """Visualizes a collection of images one by one."""
        height, width = images[0].shape[0:2]
        height = self.image_height or height
        width = self.image_width or width
        pil_images = []
        for image in images:
            if image.shape[0:2] != (height, width):
                image = resize_image(image, (width, height))
            pil_images.append(Image.fromarray(image))
        pil_images[0].save(save_path, format='GIF', save_all=True,
                           append_images=pil_images[1:],
                           duration=self.duration,
                           loop=self.loop)

    def visualize_list(self, image_list, save_path):
        """Visualizes a list of image files."""
        height, width = load_image(image_list[0]).shape[0:2]
        height = self.image_height or height
        width = self.image_width or width
        pil_images = []
        for filename in image_list:
            image = load_image(filename)
            if image.shape[0:2] != (height, width):
                image = resize_image(image, (width, height))
            pil_images.append(Image.fromarray(image))
        pil_images[0].save(save_path, format='GIF', save_all=True,
                           append_images=pil_images[1:],
                           duration=self.duration,
                           loop=self.loop)

    def visualize_directory(self, directory, save_path):
        """Visualizes all images under a directory."""
        image_list = list_images_from_dir(directory)
        self.visualize_list(image_list, save_path)
