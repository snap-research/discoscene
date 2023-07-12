# python3.7
"""Contains the class of image dataset.

`ImageDataset` is commonly used as the dataset that provides images with labels.
Concretely, each data sample (or say item) consists of an image and its
corresponding label (if provided).
"""

import numpy as np

from utils.formatting_utils import raw_label_to_one_hot
from .base_dataset import BaseDataset
from .transformations import switch_between

__all__ = ['ImageDataset']


class ImageDataset(BaseDataset):
    """Defines the image dataset class.

    NOTE: Each image can be grouped with a simple label, like 0-9 for CIFAR10,
    or 0-999 for ImageNet. The returned item format is

    {
        'index': int,
        'raw_image': np.ndarray,
        'image': np.ndarray,
        'raw_label': int,  # optional
        'label': np.ndarray  # optional
    }
    """

    def __init__(self,
                 root_dir,
                 file_format='zip',
                 annotation_path=None,
                 annotation_meta=None,
                 annotation_format='json',
                 max_samples=-1,
                 mirror=False,
                 transform_config=None,
                 transform_kwargs=None,
                 use_label=True,
                 num_classes=None):
        """Initializes the dataset.

        Args:
            use_label: Whether to enable conditioning label? Even if manually
                set this to `True`, it will be changed to `False` if labels are
                unavailable. If set to `False` manually, dataset will ignore all
                given labels. (default: True)
            num_classes: Number of classes. If not provided, the dataset will
                parse all labels to get the maximum value. This field can also
                be provided as a number larger than the actual number of
                classes. For example, sometimes, we may want to leave an
                additional class for an auxiliary task. (default: None)
        """
        super().__init__(root_dir=root_dir,
                         file_format=file_format,
                         annotation_path=annotation_path,
                         annotation_meta=annotation_meta,
                         annotation_format=annotation_format,
                         max_samples=max_samples,
                         mirror=mirror,
                         transform_config=transform_config,
                         transform_kwargs=transform_kwargs)

        self.dataset_classes = 0  # Number of classes contained in the dataset.
        self.num_classes = 0  # Actual number of classes provided by the loader.

        # Check if the dataset contains categorical information.
        self.use_label = False
        item_sample = self.items[0]
        if isinstance(item_sample, (list, tuple)) and len(item_sample) > 1:
            labels = [int(item[1]) for item in self.items]
            self.dataset_classes = max(labels) + 1
            self.use_label = use_label

        if self.use_label:
            if num_classes is None:
                self.num_classes = self.dataset_classes
            else:
                self.num_classes = int(num_classes)
            assert self.num_classes > 0
        else:
            self.num_classes = 0

    def get_raw_data(self, idx):
        # Handle data mirroring.
        do_mirror = self.mirror and idx >= (self.num_samples // 2)
        if do_mirror:
            idx = idx - self.num_samples // 2

        if self.use_label:
            image_path, raw_label = self.items[idx]
            raw_label = int(raw_label)
            label = raw_label_to_one_hot(raw_label, self.num_classes)
        else:
            image_path = self.items[idx]

        # Load image to buffer.
        buffer = np.frombuffer(self.fetch_file(image_path), dtype=np.uint8)

        idx = np.array(idx)
        do_mirror = np.array(do_mirror)
        if self.use_label:
            raw_label = np.array(raw_label)
            return [idx, do_mirror, buffer, raw_label, label]
        return [idx, do_mirror, buffer]

    @property
    def num_raw_outputs(self):
        if self.use_label:
            return 5  # [idx, do_mirror, buffer, raw_label, label]
        return 3  # [idx, do_mirror, buffer]

    def parse_transform_config(self):
        image_size = self.transform_kwargs.get('image_size')
        image_channels = self.transform_kwargs.setdefault('image_channels', 3)
        min_val = self.transform_kwargs.setdefault('min_val', -1.0)
        max_val = self.transform_kwargs.setdefault('max_val', 1.0)
        self.transform_config = dict(
            decode=dict(transform_type='Decode', image_channels=image_channels,
                        return_square=True, center_crop=True),
            resize=dict(transform_type='Resize', image_size=image_size),
            horizontal_flip=dict(transform_type='Flip',
                                 horizontal_prob=1.0,
                                 vertical_prob=0.0),
            normalize=dict(transform_type='Normalize',
                           min_val=min_val, max_val=max_val)
        )

    def transform(self, raw_data, use_dali=False):
        if self.use_label:
            idx, do_mirror, buffer, raw_label, label = raw_data
        else:
            idx, do_mirror, buffer = raw_data
        # TODO need to be check
        op_list = [key for key in self.transforms.keys() if key not in ['decode', 'horizontal_flip', 'normalize']]
        raw_image = self.transforms['decode'](buffer, use_dali=use_dali)
        raw_image = raw_image[:, :, :3]
        for op in op_list:
            raw_image = self.transforms[op](raw_image, use_dali=use_dali)
        # raw_image = self.transforms['resize1'](raw_image, use_dali=use_dali)
        # raw_image = self.transforms['centercrop'](raw_image, use_dali=use_dali)
        # raw_image = self.transforms['resize2'](raw_image, use_dali=use_dali)
        flipped_image = self.transforms['horizontal_flip'](
            raw_image, use_dali=use_dali)
        image = switch_between(cond=do_mirror,
                               cond_true=flipped_image,
                               cond_false=raw_image,
                               use_dali=use_dali)
        image = self.transforms['normalize'](image, use_dali=use_dali)

        if self.use_label:
            return [idx, raw_image, image, raw_label, label]
        return [idx, raw_image, image]

    @property
    def output_keys(self):
        if self.use_label:
            return ['index', 'raw_image', 'image', 'raw_label', 'label']
        return ['index', 'raw_image', 'image']

    def info(self):
        dataset_info = super().info()
        dataset_info['Dataset classes'] = self.dataset_classes
        dataset_info['Use label'] = self.use_label
        if self.use_label:
            dataset_info['Num classes for training'] = self.num_classes
        return dataset_info
