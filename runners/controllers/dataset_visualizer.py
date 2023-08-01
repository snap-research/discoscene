# python3.7
"""Contains the running controller to visualize the training dataset."""

import os.path
import numpy as np

from utils.visualizers import GridVisualizer
from utils.image_utils import postprocess_image
from .base_controller import BaseController

__all__ = ['DatasetVisualizer']


class DatasetVisualizer(BaseController):
    """Defines the running controller to visualize the training dataset.

    This controller is primarily designed to visualize the training dataset at
    the start of training. It can be used to check the data is formatted
    properly, including paired data, categorical data, etc.

    NOTE: The controller is set to `MEDIUM` priority, and will only be executed
    on the chief worker. Also, this controller will only be executed ONCE,
    at the beginning of the entire training.

    Visualization settings:

    - viz_keys: List of keys in `runner.train_loader`, which will be visualized
        in order.
    - viz_num: Number of batch items (regarding one category) to visualize.
        If negative, minibatch will be used. (default: -1)
    - viz_name: Name of the visualization, which will both used as the name of
        the file to save and display in TensorBoard.
        (default: `Training Dataset`)
    - viz_groups: Number of visualization groups, which is specially used to
        control how many items to display on each row (or column if not
        `row-major`). For example, `viz_groups = 1` means all items (regarding
        one category) will be visualized on the same row. `viz_groups = 2` means
        all items (regarding one category) will be evenly distributed to two
        rows. For a better visualization, it is recommended to use
        `viz_groups = 1` for the categorical case. (default: 1)
    - viz_classes: Maximum number of categories to visualize. This field is
        particularly used to avoid there are too many classes in the dataset,
        such as ImageNet. If this field is set, the first `viz_classes`
        categories will be visualized. Non-positive means no limitation on the
        number of categories for visualization. (default: -1)
    - row_major: Whether to visualize the batch in the row-major format, where
        each column represents a particular item. (default: True)
    - min_val: Minimum pixel value of the images to visualize. This field is
        used to post-process the images back to pixel range [0, 255].
        (default: -1.0)
    - max_val: Maximum pixel value of the images to visualize. This field is
        used to post-process the images back to pixel range [0, 255].
        (default: 1.0)
    - shuffle: Whether to shuffle the dataset before visualization.
        (default: False)
    """

    def __init__(self, config):
        assert isinstance(config, dict)
        config.setdefault('priority', 'MEDIUM')
        config.setdefault('every_n_iters', -1)
        config.setdefault('every_n_epochs', -1)
        config.setdefault('first_iter', True)
        config.setdefault('last_iter', False)
        config.setdefault('chief_only', True)
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
        self.viz_classes = config.get('viz_classes', -1)
        self.row_major = config.get('row_major', True)
        self.min_val = config.get('min_val', -1.0)
        self.max_val = config.get('max_val', 1.0)
        self.shuffle = config.get('shuffle', False)

        self._visualizer = GridVisualizer()

    def setup(self, runner):
        dataset = runner.train_loader.dataset
        if hasattr(dataset, 'use_label') and dataset.use_label:
            num_classes = dataset.num_classes
        else:
            num_classes = 1
        if self.viz_classes > 0:
            self.viz_classes = min(self.viz_classes, num_classes)
        else:
            self.viz_classes = num_classes

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
        runner.logger.info(f'Categorical visualization: {self.viz_classes}',
                           indent_level=3)
        runner.logger.info(f'Minimum image pixel value: {self.min_val}',
                           indent_level=3)
        runner.logger.info(f'Maximum image pixel value: {self.max_val}',
                           indent_level=3)
        runner.logger.info(f'Shuffle: {self.shuffle}', indent_level=3)
        super().setup(runner)

    def execute_before_iteration(self, runner):
        if not self.viz_keys or self.viz_num == 0:
            return

        # Parse visualization num.
        viz_num = self.viz_num if self.viz_num >= 0 else runner.minibatch
        num_per_class = [viz_num for _ in range(self.viz_classes)]
        total_num = viz_num * self.viz_classes

        # Reset visualization grid.
        num_keys = len(self.viz_keys)
        grid_rows = num_keys * self.viz_groups * self.viz_classes
        grid_cols = int(viz_num / self.viz_groups + 0.5)
        if not self.row_major:
            grid_rows, grid_cols = grid_cols, grid_rows
        self._visualizer.reset(num_rows=grid_rows, num_cols=grid_cols)

        # Place images for visualization.
        dataset = runner.train_loader.dataset
        order = np.arange(len(dataset))
        if self.shuffle:
            order = np.random.permutation(order)
        sample_idx = 0
        while total_num > 0:
            # Fetch data for visualization.
            sample = dataset[order[sample_idx]]
            sample_idx += 1
            # Get compatible with categorical visualization.
            if self.viz_classes == 1:
                sample['raw_label'] = 0  # pseudo label.
            if 'raw_label' not in sample:
                raise ValueError('Key `raw_label` is required for categorical '
                                 'visualization!')
            label = sample['raw_label']
            if label >= self.viz_classes:  # categories not to visualize.
                continue
            if num_per_class[label] == 0:  # categories with enough samples.
                continue
            # Visualize.
            image_idx = viz_num - num_per_class[label]
            num_per_class[label] -= 1
            total_num -= 1
            for key_idx, key in enumerate(self.viz_keys):
                image = sample[key]
                assert isinstance(image, np.ndarray), 'Invalid image type!'
                try:
                    assert image.ndim == 3, 'Invalid number of image dimensions!'
                except:
                    import ipdb;ipdb.set_trace()
                if image.dtype != np.uint8:
                    # CHW to HWC
                    image = postprocess_image(image[None],
                                              min_val=self.min_val,
                                              max_val=self.max_val)[0]
                if self.row_major:
                    row_idx = key_idx + num_keys * (image_idx // grid_cols)
                    row_idx = row_idx + num_keys * self.viz_groups * label
                    col_idx = image_idx % grid_cols
                else:
                    row_idx = image_idx % grid_rows
                    col_idx = key_idx + num_keys * (image_idx // grid_rows)
                    col_idx = col_idx + num_keys * self.viz_groups * label
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
