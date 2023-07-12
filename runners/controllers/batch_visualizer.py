# python3.7
"""Contains the running controller to visualize training batch."""

import os.path

import torch
import torch.distributed as dist

from utils.visualizers import GridVisualizer
from utils.image_utils import postprocess_image
from .base_controller import BaseController

__all__ = ['BatchVisualizer']


class BatchVisualizer(BaseController):
    """Defines the running controller to visualize the training batch.

    This controller is primarily designed to visualize the samples (images) of
    a particular batch. There can be more than one image within a certain batch
    item (e.g., paired data in pix2pix). This controller supports visualizing
    them together.

    NOTE: The controller is set to `MEDIUM` priority.

    Visualization settings:

    - viz_keys: List of keys in `runner.batch_data`, which will be visualized
        in order.
    - viz_num: Number of batch items to visualize. If negative, all items on one
        GPU will be used. (default: -1)
    - viz_name: Name of the visualization, which will both used as the name of
        the file to save and display in TensorBoard. (default: `Training Batch`)
    - viz_groups: Number of visualization groups, which is specially used to
        control how many items to display on each row (or column if not
        `row-major`). For example, `viz_groups = 1` means all items will be
        visualized on the same row, while `viz_groups = 2` means all items will
        be evenly distributed to two rows. (default: 1)
    - row_major: Whether to visualize the batch in the row-major format, where
        each column represents a particular item. (default: True)
    - min_val: Minimum pixel value of the images to visualize. This field is
        used to post-process the images back to pixel range [0, 255].
        (default: -1.0)
    - max_val: Maximum pixel value of the images to visualize. This field is
        used to post-process the images back to pixel range [0, 255].
        (default: 1.0)
    - image_channels: image channel number of a grid. It is the maximum image 
        channel number of images you want to visualize in the same grid. For
        example, if you want to visualize RGB and RGBA images in the same grid,
        this number should be 4.
        (default: 3)
    - use_black_background: Whether to use black background. It will initialize
        the grid to all zero image if this flag is set True, otherwise 255s.
        (default: True)
    """

    def __init__(self, config):
        assert isinstance(config, dict)
        config.setdefault('priority', 'MEDIUM')
        super().__init__(config)

        # Visualization options.
        viz_keys = config.get('viz_keys', [])
        if not isinstance(viz_keys, (tuple, list)):
            viz_keys = [viz_keys]
        else:
            viz_keys = list(viz_keys)
        self.viz_keys = viz_keys
        self.viz_num = config.get('viz_num', -1)
        self.viz_name = config.get('viz_name', 'Training Batch')
        self.viz_groups = config.get('viz_groups', 1)
        self.row_major = config.get('row_major', True)
        self.min_val = config.get('min_val', -1.0)
        self.max_val = config.get('max_val', 1.0)

        self._visualizer = GridVisualizer(
            image_channels=config.get('image_channels', 3),
            use_black_background=config.get('use_black_background', False))

    def setup(self, runner):
        runner.logger.info('Visualization settings:', indent_level=2)
        runner.logger.info(f'Visualization keys: {self.viz_keys}',
                           indent_level=3)
        runner.logger.info(f'Visualization num: {self.viz_num}',
                           indent_level=3)
        runner.logger.info(f'Visualization name: {self.viz_name}',
                           indent_level=3)
        runner.logger.info(f'Visualization groups: {self.viz_groups}',
                           indent_level=3)
        viz_format = 'row-major' if self.row_major else 'column-major'
        runner.logger.info(f'Visualization format: {viz_format}',
                           indent_level=3)
        runner.logger.info(f'Minimum image pixel value: {self.min_val}',
                           indent_level=3)
        runner.logger.info(f'Maximum image pixel value: {self.max_val}',
                           indent_level=3)
        super().setup(runner)

    def execute_after_iteration(self, runner):
        if not self.viz_keys or self.viz_num == 0:
            return

        # Parse visualization num.
        if self.viz_num < 0:
            viz_num = runner.batch_size
        else:
            viz_num = min(self.viz_num, runner.minibatch)

        # Gather results across GPUs if needed.
        batch_data = {key: [] for key in self.viz_keys}
        if viz_num <= runner.batch_size:
            if runner.is_chief:
                for key in self.viz_keys:
                    batch_data[key] = runner.batch_data[key]
        else:
            for key in self.viz_keys:
                for _ in range(runner.world_size):
                    batch_data[key].append(
                        torch.zeros_like(runner.batch_data[key]))
                dist.all_gather(batch_data[key], runner.batch_data[key])
                batch_data[key] = torch.cat(batch_data[key], dim=0)

        # Only chief executes visualization.
        if not runner.is_chief:
            del batch_data  # save memory
            dist.barrier()  # synchronize across workers
            return

        # Reset visualization grid.
        num_keys = len(self.viz_keys)
        grid_rows = num_keys * self.viz_groups
        grid_cols = int(viz_num / self.viz_groups + 0.5)
        if not self.row_major:
            grid_rows, grid_cols = grid_cols, grid_rows
        self._visualizer.reset(num_rows=grid_rows, num_cols=grid_cols)

        # Place images for visualization.
        for key_idx, images in enumerate(batch_data.values()):
            assert images.shape[0] >= viz_num
            images = postprocess_image(images.detach().cpu().numpy(),
                                       min_val=self.min_val,
                                       max_val=self.max_val)
            for image_idx, image in enumerate(images[:viz_num]):
                if self.row_major:
                    row_idx = key_idx + num_keys * (image_idx // grid_cols)
                    col_idx = image_idx % grid_cols
                else:
                    row_idx = image_idx % grid_rows
                    col_idx = key_idx + num_keys * (image_idx // grid_rows)
                self._visualizer.add(row_idx, col_idx, image)

        # Save result.
        save_name = f'{self.viz_name}-{runner.iter:06d}.png'
        save_name = save_name.lower().replace(' ', '_')
        save_path = os.path.join(runner.result_dir, save_name)
        self._visualizer.save(save_path)
        if runner.tb_writer is not None:
            runner.tb_writer.add_image(self.viz_name,
                                       self._visualizer.grid,
                                       global_step=runner.iter,
                                       dataformats='HWC')
            runner.tb_writer.flush()

        # Synchronize across workers.
        dist.barrier()
