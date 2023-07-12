# python3.7
"""Contains the visualizer to visualize images by composing them as a gird."""

from ..image_utils import get_blank_image
from ..image_utils import get_grid_shape
from ..image_utils import parse_image_size
from ..image_utils import load_image
from ..image_utils import save_image
from ..image_utils import resize_image
from ..image_utils import list_images_from_dir

__all__ = ['GridVisualizer']


class GridVisualizer(object):
    """Defines the visualizer that visualizes images as a grid.

    Basically, given a collection of images, this visualizer stitches them one
    by one. Notably, this class also supports adding spaces between images,
    adding borders around images, and using white/black background.

    Example:

    grid = GridVisualizer(num_rows, num_cols)
    for i in range(num_rows):
        for j in range(num_cols):
            grid.add(i, j, image)
    grid.save('visualize.jpg')
    """

    def __init__(self,
                 grid_size=0,
                 num_rows=0,
                 num_cols=0,
                 is_portrait=False,
                 image_size=None,
                 image_channels=0,
                 row_spacing=0,
                 col_spacing=0,
                 border_left=0,
                 border_right=0,
                 border_top=0,
                 border_bottom=0,
                 use_black_background=True):
        """Initializes the grid visualizer.

        Args:
            grid_size: Total number of cells, i.e., height * width. (default: 0)
            num_rows: Number of rows. (default: 0)
            num_cols: Number of columns. (default: 0)
            is_portrait: Whether the grid should be portrait or landscape.
                This is only used when it requires to compute `num_rows` and
                `num_cols` automatically. See function `get_grid_shape()` in
                file `./image_utils.py` for details. (default: False)
            image_size: Size to visualize each image. (default: 0)
            image_channels: Number of image channels. (default: 0)
            row_spacing: Spacing between rows. (default: 0)
            col_spacing: Spacing between columns. (default: 0)
            border_left: Width of left border. (default: 0)
            border_right: Width of right border. (default: 0)
            border_top: Width of top border. (default: 0)
            border_bottom: Width of bottom border. (default: 0)
            use_black_background: Whether to use black background.
                (default: True)
        """
        self.reset(grid_size, num_rows, num_cols, is_portrait)
        self.set_image_size(image_size)
        self.set_image_channels(image_channels)
        self.set_row_spacing(row_spacing)
        self.set_col_spacing(col_spacing)
        self.set_border_left(border_left)
        self.set_border_right(border_right)
        self.set_border_top(border_top)
        self.set_border_bottom(border_bottom)
        self.set_background(use_black_background)
        self.grid = None

    def reset(self,
              grid_size=0,
              num_rows=0,
              num_cols=0,
              is_portrait=False):
        """Resets the grid shape, i.e., number of rows/columns."""
        if grid_size > 0:
            num_rows, num_cols = get_grid_shape(grid_size,
                                                height=num_rows,
                                                width=num_cols,
                                                is_portrait=is_portrait)
        self.grid_size = num_rows * num_cols
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.grid = None

    def set_image_size(self, image_size=None):
        """Sets the image size of each cell in the grid."""
        height, width = parse_image_size(image_size)
        self.image_height = height
        self.image_width = width

    def set_image_channels(self, image_channels=0):
        """Sets the number of channels of the grid."""
        self.image_channels = image_channels

    def set_row_spacing(self, row_spacing=0):
        """Sets the spacing between grid rows."""
        self.row_spacing = row_spacing

    def set_col_spacing(self, col_spacing=0):
        """Sets the spacing between grid columns."""
        self.col_spacing = col_spacing

    def set_border_left(self, border_left=0):
        """Sets the width of the left border of the grid."""
        self.border_left = border_left

    def set_border_right(self, border_right=0):
        """Sets the width of the right border of the grid."""
        self.border_right = border_right

    def set_border_top(self, border_top=0):
        """Sets the width of the top border of the grid."""
        self.border_top = border_top

    def set_border_bottom(self, border_bottom=0):
        """Sets the width of the bottom border of the grid."""
        self.border_bottom = border_bottom

    def set_background(self, use_black=True):
        """Sets the grid background."""
        self.use_black_background = use_black

    def init_grid(self):
        """Initializes the grid with a blank image."""
        assert self.num_rows > 0
        assert self.num_cols > 0
        assert self.image_height > 0
        assert self.image_width > 0
        assert self.image_channels > 0
        grid_height = (self.image_height * self.num_rows +
                       self.row_spacing * (self.num_rows - 1) +
                       self.border_top + self.border_bottom)
        grid_width = (self.image_width * self.num_cols +
                      self.col_spacing * (self.num_cols - 1) +
                      self.border_left + self.border_right)
        self.grid = get_blank_image(grid_height, grid_width,
                                    channels=self.image_channels,
                                    use_black=self.use_black_background)

    def add(self, i, j, image):
        """Adds an image into the grid.

        NOTE: The input image is assumed to be with `RGB` channel order.
        """
        channels = 1 if image.ndim == 2 else image.shape[2]
        if self.grid is None:
            height, width = image.shape[0:2]
            height = self.image_height or height
            width = self.image_width or width
            channels = self.image_channels or channels
            self.set_image_size((height, width))
            self.set_image_channels(channels)
            self.init_grid()
        if image.shape[0:2] != (self.image_height, self.image_width):
            image = resize_image(image, (self.image_width, self.image_height))
        y = self.border_top + i * (self.image_height + self.row_spacing)
        x = self.border_left + j * (self.image_width + self.col_spacing)
        self.grid[y:y + self.image_height,
                  x:x + self.image_width,
                  :channels] = image

    def visualize_collection(self,
                             images,
                             save_path=None,
                             num_rows=0,
                             num_cols=0,
                             is_portrait=False,
                             is_row_major=True):
        """Visualizes a collection of images one by one."""
        self.grid = None
        self.reset(grid_size=len(images),
                   num_rows=num_rows,
                   num_cols=num_cols,
                   is_portrait=is_portrait)
        for idx, image in enumerate(images):
            if is_row_major:
                row_idx, col_idx = divmod(idx, self.num_cols)
            else:
                col_idx, row_idx = divmod(idx, self.num_rows)
            self.add(row_idx, col_idx, image)
        if save_path:
            self.save(save_path)

    def visualize_list(self,
                       image_list,
                       save_path=None,
                       num_rows=0,
                       num_cols=0,
                       is_portrait=False,
                       is_row_major=True):
        """Visualizes a list of image files."""
        self.grid = None
        self.reset(grid_size=len(image_list),
                   num_rows=num_rows,
                   num_cols=num_cols,
                   is_portrait=is_portrait)
        for idx, filename in enumerate(image_list):
            image = load_image(filename)
            if is_row_major:
                row_idx, col_idx = divmod(idx, self.num_cols)
            else:
                col_idx, row_idx = divmod(idx, self.num_rows)
            self.add(row_idx, col_idx, image)
        if save_path:
            self.save(save_path)

    def visualize_directory(self,
                            directory,
                            save_path=None,
                            num_rows=0,
                            num_cols=0,
                            is_portrait=False,
                            is_row_major=True):
        """Visualizes all images under a directory."""
        image_list = list_images_from_dir(directory)
        self.visualize_list(image_list=image_list,
                            save_path=save_path,
                            num_rows=num_rows,
                            num_cols=num_cols,
                            is_portrait=is_portrait,
                            is_row_major=is_row_major)

    def save(self, path):
        """Saves the grid."""
        save_image(path, self.grid)
