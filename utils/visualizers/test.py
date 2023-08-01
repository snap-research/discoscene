# python3.7
"""Unit test for visualizer."""

import os
import skvideo.datasets

from ..image_utils import save_image
from . import GridVisualizer
from . import HtmlVisualizer
from . import HtmlReader
from . import GifVisualizer
from . import VideoVisualizer
from . import VideoReader

__all__ = ['test_visualizer']

_TEST_DIR = 'visualizer_test'


def test_visualizer(test_dir=_TEST_DIR):
    """Tests visualizers."""
    print('========== Start Visualizer Test ==========')

    frame_dir = os.path.join(test_dir, 'test_frames')
    os.makedirs(frame_dir, exist_ok=True)

    print('===== Testing `VideoReader` =====')
    # Total 132 frames, with size (720, 1080).
    video_reader = VideoReader(skvideo.datasets.bigbuckbunny())
    frame_height = video_reader.frame_height
    frame_width = video_reader.frame_width
    frame_size = (frame_height, frame_width)
    half_size = (frame_height // 2, frame_width // 2)
    # Save frames as the test set.
    for idx in range(80):
        frame = video_reader.read()
        save_image(os.path.join(frame_dir, f'{idx:02d}.png'), frame)

    print('===== Testing `GirdVisualizer` =====')
    grid_visualizer = GridVisualizer()
    grid_visualizer.set_row_spacing(30)
    grid_visualizer.set_col_spacing(30)
    grid_visualizer.set_background(use_black=True)
    path = os.path.join(test_dir, 'portrait_row_major_ori_space30_black.png')
    grid_visualizer.visualize_directory(frame_dir, path,
                                        is_portrait=True, is_row_major=True)
    path = os.path.join(
        test_dir, 'landscape_col_major_downsample_space15_white.png')
    grid_visualizer.set_image_size(half_size)
    grid_visualizer.set_row_spacing(15)
    grid_visualizer.set_col_spacing(15)
    grid_visualizer.set_background(use_black=False)
    grid_visualizer.visualize_directory(frame_dir, path,
                                        is_portrait=False, is_row_major=False)

    print('===== Testing `HtmlVisualizer` =====')
    html_visualizer = HtmlVisualizer()
    path = os.path.join(test_dir, 'portrait_col_major_ori.html')
    html_visualizer.visualize_directory(frame_dir, path,
                                        is_portrait=True, is_row_major=False)
    path = os.path.join(test_dir, 'landscape_row_major_downsample.html')
    html_visualizer.set_image_size(half_size)
    html_visualizer.visualize_directory(frame_dir, path,
                                        is_portrait=False, is_row_major=True)

    print('===== Testing `HtmlReader` =====')
    path = os.path.join(test_dir, 'landscape_row_major_downsample.html')
    html_reader = HtmlReader(path)
    for j in range(html_reader.num_cols):
        assert html_reader.get_header(j) == ''
    parsed_dir = os.path.join(test_dir, 'parsed_frames')
    os.makedirs(parsed_dir, exist_ok=True)
    for i in range(html_reader.num_rows):
        for j in range(html_reader.num_cols):
            idx = i * html_reader.num_cols + j
            assert html_reader.get_text(i, j).endswith(f'(index {idx:03d})')
            image = html_reader.get_image(i, j, image_size=frame_size)
            assert image.shape[0:2] == frame_size
            save_image(os.path.join(parsed_dir, f'{idx:02d}.png'), image)

    print('===== Testing `GifVisualizer` =====')
    gif_visualizer = GifVisualizer()
    path = os.path.join(test_dir, 'gif_ori.gif')
    gif_visualizer.visualize_directory(frame_dir, path)
    gif_visualizer.set_image_size(half_size)
    path = os.path.join(test_dir, 'gif_downsample.gif')
    gif_visualizer.visualize_directory(frame_dir, path)

    print('===== Testing `VideoVisualizer` =====')
    video_visualizer = VideoVisualizer()
    path = os.path.join(test_dir, 'video_ori.mp4')
    video_visualizer.visualize_directory(frame_dir, path)
    path = os.path.join(test_dir, 'video_downsample.mp4')
    video_visualizer.set_frame_size(half_size)
    video_visualizer.visualize_directory(frame_dir, path)

    print('========== Finish Visualizer Test ==========')
