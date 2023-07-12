# python3.7
"""Contains the visualizer to visualize images with HTML page."""

import os
import base64
import cv2
import numpy as np
from bs4 import BeautifulSoup

from ..image_utils import get_grid_shape
from ..image_utils import parse_image_size
from ..image_utils import load_image
from ..image_utils import resize_image
from ..image_utils import list_images_from_dir

__all__ = ['HtmlVisualizer', 'HtmlReader']


def get_sortable_html_header(column_name_list, sort_by_ascending=False):
    """Gets header for sortable HTML page.

    Basically, the HTML page contains a sortable table, where user can sort the
    rows by a particular column by clicking the column head.

    Example:

    column_name_list = [name_1, name_2, name_3]
    header = get_sortable_html_header(column_name_list)
    footer = get_sortable_html_footer()
    sortable_table = ...
    html_page = header + sortable_table + footer

    Args:
        column_name_list: List of column header names.
        sort_by_ascending: Default sorting order. If set as `True`, the HTML
            page will be sorted by ascending order when the header is clicked
            for the first time.

    Returns:
        A string, which represents for the header for a sortable HTML page.
    """
    header = '\n'.join([
        '<script type="text/javascript">',
        'var column_idx;',
        'var sort_by_ascending = ' + str(sort_by_ascending).lower() + ';',
        '',
        'function sorting(tbody, column_idx){',
        '    this.column_idx = column_idx;',
        '    Array.from(tbody.rows)',
        '             .sort(compareCells)',
        '             .forEach(function(row) { tbody.appendChild(row); })',
        '    sort_by_ascending = !sort_by_ascending;',
        '}',
        '',
        'function compareCells(row_a, row_b) {',
        '    var val_a = row_a.cells[column_idx].innerText;',
        '    var val_b = row_b.cells[column_idx].innerText;',
        '    var flag = sort_by_ascending ? 1 : -1;',
        '    return flag * (val_a > val_b ? 1 : -1);',
        '}',
        '</script>',
        '',
        '<html>',
        '',
        '<head>',
        '<style>',
        '    table {',
        '        border-spacing: 0;',
        '        border: 1px solid black;',
        '    }',
        '    th {',
        '        cursor: pointer;',
        '    }',
        '    th, td {',
        '        text-align: left;',
        '        vertical-align: middle;',
        '        border-collapse: collapse;',
        '        border: 0.5px solid black;',
        '        padding: 8px;',
        '    }',
        '    tr:nth-child(even) {',
        '        background-color: #d2d2d2;',
        '    }',
        '</style>',
        '</head>',
        '',
        '<body>',
        '',
        '<table>',
        '<thead>',
        '<tr>',
        ''])
    for idx, name in enumerate(column_name_list):
        header += f'    <th onclick="sorting(tbody, {idx})">{name}</th>\n'
    header += '</tr>\n'
    header += '</thead>\n'
    header += '<tbody id="tbody">\n'

    return header


def get_sortable_html_footer():
    """Gets footer for sortable HTML page.

    Check function `get_sortable_html_header()` for more details.
    """
    return '</tbody>\n</table>\n\n</body>\n</html>\n'


def encode_image_to_html_str(image, image_size=None):
    """Encodes an image to HTML language.

    NOTE: Input image is always assumed to be with `RGB` channel order.

    Args:
        image: The input image to encode. Should be with `RGB` channel order.
        image_size: This field is used to resize the image before encoding.
            `None` disables resizing. (default: None)

    Returns:
        A string that represents the encoded image.
    """
    if image is None:
        return ''

    assert image.ndim == 3 and image.shape[2] in [1, 3, 4]
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)

    # Resize the image if needed.
    height, width = parse_image_size(image_size)
    height = height or image.shape[0]
    width = width or image.shape[1]
    if image.shape[0:2] != (height, width):
        image = resize_image(image, (width, height))

    # Encode the image to HTML-format string.
    if image.shape[2] == 4:  # Use `png` to encoder RGBA image.
        encoded = cv2.imencode('.png', image)[1].tostring()
        encoded_base64 = base64.b64encode(encoded).decode('utf-8')
        html_str = f'<img src="data:image/png;base64, {encoded_base64}"/>'
    else:
        encoded = cv2.imencode('.jpg', image)[1].tostring()
        encoded_base64 = base64.b64encode(encoded).decode('utf-8')
        html_str = f'<img src="data:image/jpeg;base64, {encoded_base64}"/>'

    return html_str


def decode_html_str_to_image(html_str, image_size=None):
    """Decodes an image from HTML string.

    Args:
        html_str: An HTML string that represents an image.
        image_size: This field is used to resize the image after decoding.
            `None` disables resizing. (default: None)

    Returns:
        An image with `RGB` channel order.
    """
    if not html_str:
        return None

    assert isinstance(html_str, str)
    image_str = html_str.split(',')[-1].strip()
    encoded_image = base64.b64decode(image_str)
    encoded_image_numpy = np.frombuffer(encoded_image, dtype=np.uint8)
    image = cv2.imdecode(encoded_image_numpy, flags=cv2.IMREAD_UNCHANGED)

    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    assert image.ndim == 3 and image.shape[2] in [1, 3, 4]
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

    # Resize the image if needed.
    height, width = parse_image_size(image_size)
    height = height or image.shape[0]
    width = width or image.shape[1]
    if image.shape[0:2] != (height, width):
        image = resize_image(image, (width, height))

    return image


class HtmlVisualizer(object):
    """Defines the HTML visualizer that visualizes images on an HTML page.

    This class can be used to visualize image results on an HTML page.
    Basically, it is based on an HTML-format sorted table with helper functions
    `get_sortable_html_header()`, `get_sortable_html_footer()`, and
    `encode_image_to_html_str()`. To simplify the usage, specifying the
    following fields are enough to create a visualization page:

    (1) num_rows: Number of rows of the table (header-row exclusive).
    (2) num_cols: Number of columns of the table.
    (3) header_contents (optional): Title of each column.

    NOTE: `grid_size` can be used to assign `num_rows` and `num_cols`
    automatically.

    Example:

    html = HtmlVisualizer(num_rows, num_cols)
    html.set_headers([...])
    for i in range(num_rows):
        for j in range(num_cols):
            html.set_cell(i, j, text=..., image=..., highlight=False)
    html.save('visualize.html')
    """

    def __init__(self,
                 grid_size=0,
                 num_rows=0,
                 num_cols=0,
                 is_portrait=True,
                 image_size=None):
        """Initializes the html visualizer.

        Args:
            grid_size: Total number of cells, i.e., height * width. (default: 0)
            num_rows: Number of rows. (default: 0)
            num_cols: Number of columns. (default: 0)
            is_portrait: Whether the HTML page should be portrait or landscape.
                This is only used when it requires to compute `num_rows` and
                `num_cols` automatically. See function `get_grid_shape()` in
                file `./image_utils.py` for details. (default: True)
            image_size: Size to visualize each image. (default: None)
        """
        self.reset(grid_size, num_rows, num_cols, is_portrait)
        self.set_image_size(image_size)

    def reset(self,
              grid_size=0,
              num_rows=0,
              num_cols=0,
              is_portrait=True):
        """Resets the HTML page with new number of rows and columns."""
        if grid_size > 0:
            num_rows, num_cols = get_grid_shape(grid_size,
                                                height=num_rows,
                                                width=num_cols,
                                                is_portrait=is_portrait)
        self.grid_size = num_rows * num_cols
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.headers = ['' for _ in range(self.num_cols)]
        self.cells = [[{
            'text': '',
            'image': '',
            'highlight': False,
        } for _ in range(self.num_cols)] for _ in range(self.num_rows)]

    def set_image_size(self, image_size=None):
        """Sets the image size of each cell in the HTML page."""
        self.image_size = image_size

    def set_header(self, col_idx, content):
        """Sets the content of a particular header by column index."""
        self.headers[col_idx] = content

    def set_headers(self, contents):
        """Sets the contents of all headers."""
        assert isinstance(contents, (list, tuple))
        assert len(contents) == self.num_cols
        for col_idx, content in enumerate(contents):
            self.set_header(col_idx, content)

    def set_cell(self, row_idx, col_idx, text='', image=None, highlight=False):
        """Sets the content of a particular cell.

        Basically, a cell contains some text as well as an image. Both text and
        image can be empty.

        NOTE: The image is assumed to be with `RGB` channel order.

        Args:
            row_idx: Row index of the cell to edit.
            col_idx: Column index of the cell to edit.
            text: Text to add into the target cell. (default: None)
            image: Image to show in the target cell. Should be with `RGB`
                channel order. (default: None)
            highlight: Whether to highlight this cell. (default: False)
        """
        self.cells[row_idx][col_idx]['text'] = text
        self.cells[row_idx][col_idx]['image'] = encode_image_to_html_str(
            image, self.image_size)
        self.cells[row_idx][col_idx]['highlight'] = bool(highlight)

    def visualize_collection(self,
                             images,
                             save_path=None,
                             num_rows=0,
                             num_cols=0,
                             is_portrait=True,
                             is_row_major=True):
        """Visualizes a collection of images one by one."""
        self.reset(grid_size=len(images),
                   num_rows=num_rows,
                   num_cols=num_cols,
                   is_portrait=is_portrait)
        for idx, image in enumerate(images):
            if is_row_major:
                row_idx, col_idx = divmod(idx, self.num_cols)
            else:
                col_idx, row_idx = divmod(idx, self.num_rows)
            self.set_cell(row_idx, col_idx, text=f'Index {idx:03d}',
                          image=image)
        if save_path:
            self.save(save_path)

    def visualize_list(self,
                       image_list,
                       save_path=None,
                       num_rows=0,
                       num_cols=0,
                       is_portrait=True,
                       is_row_major=True):
        """Visualizes a list of image files."""
        self.reset(grid_size=len(image_list),
                   num_rows=num_rows,
                   num_cols=num_cols,
                   is_portrait=is_portrait)
        for idx, filename in enumerate(image_list):
            basename = os.path.basename(filename)
            image = load_image(filename)
            if is_row_major:
                row_idx, col_idx = divmod(idx, self.num_cols)
            else:
                col_idx, row_idx = divmod(idx, self.num_rows)
            self.set_cell(row_idx, col_idx,
                          text=f'{basename} (index {idx:03d})', image=image)
        if save_path:
            self.save(save_path)

    def visualize_directory(self,
                            directory,
                            save_path=None,
                            num_rows=0,
                            num_cols=0,
                            is_portrait=True,
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
        """Saves the HTML page."""
        html = ''
        for i in range(self.num_rows):
            html += '<tr>\n'
            for j in range(self.num_cols):
                text = self.cells[i][j]['text']
                image = self.cells[i][j]['image']
                if self.cells[i][j]['highlight']:
                    color = ' bgcolor="#FF8888"'
                else:
                    color = ''
                if text:
                    html += f'    <td{color}>{text}<br><br>{image}</td>\n'
                else:
                    html += f'    <td{color}>{image}</td>\n'
            html += '</tr>\n'

        header = get_sortable_html_header(self.headers)
        footer = get_sortable_html_footer()

        with open(path, 'w') as f:
            f.write(header + html + footer)


class HtmlReader(object):
    """Defines the HTML page reader.

    This class can be used to parse results from the visualization page
    generated by `HtmlVisualizer`.

    Example:

    html = HtmlReader(html_path)
    for j in range(html.num_cols):
        header = html.get_header(j)
    for i in range(html.num_rows):
        for j in range(html.num_cols):
            text = html.get_text(i, j)
            image = html.get_image(i, j, image_size=None)
    """
    def __init__(self, path):
        """Initializes by loading the content from file."""
        self.path = path

        # Load content.
        with open(path, 'r') as f:
            self.html = BeautifulSoup(f, 'html.parser')

        # Parse headers.
        thead = self.html.find('thead')
        headers = thead.findAll('th')
        self.headers = []
        for header in headers:
            self.headers.append(header.text)
        self.num_cols = len(self.headers)

        # Parse cells.
        tbody = self.html.find('tbody')
        rows = tbody.findAll('tr')
        self.cells = []
        for row in rows:
            cells = row.findAll('td')
            self.cells.append([])
            for cell in cells:
                self.cells[-1].append({
                    'text': cell.text,
                    'image': cell.find('img')['src'],
                })
            assert len(self.cells[-1]) == self.num_cols
        self.num_rows = len(self.cells)

    def get_header(self, j):
        """Gets header for a particular column."""
        return self.headers[j]

    def get_text(self, i, j):
        """Gets text from a particular cell."""
        return self.cells[i][j]['text']

    def get_image(self, i, j, image_size=None):
        """Gets image from a particular cell."""
        return decode_html_str_to_image(self.cells[i][j]['image'], image_size)
