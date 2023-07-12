# python3.7
"""Contains the base class of dataset.

Each dataset establishes a map from indices to data items (or say samples). Each
data item consists of an image (or a set of associated images, like paired
images for pix2pix) and the corresponding annotations. In this way, it is
convenient for data loaders to directly get an item by its index. Besides
getting the item, the dataset should also handle data pre-processing (i.e., data
transformation), such that the data received by the data loader can be directly
grouped to batches and further used for training.

NOTE: Dataset supports reading raw data from disk, parsing annotations, and
pre-processing the raw data. But it merely handles one data item (instead of a
batch) at one time.
"""

import os.path
import json
import pickle

from torch.utils.data import Dataset

from utils.misc import parse_file_format, parse_ann_format
from .file_readers import build_file_reader
from .transformations import build_transformation

__all__ = ['BaseDataset']

_ALLOWED_ANNO_FORMATS = ['json', 'txt', 'pkl', 'None']


class BaseDataset(Dataset):
    """Defines the base dataset class.

    It is possible to get an data item (which is finally fed into the model)
    with `dataset[index]`.

    Common functions:

    (1) __len__(): Return the number of items contained in the dataset. If the
        dataset is mirrored, i.e., with `self.mirror = True`, this field doubles
        the actual number of items.
    (2) fetch_file(): Fetch a particular file from disk.
    (3) build_transformations(): Initialize each transformation node within the
        data pre-processing pipeline with given configuration.
    (4) __getitem__(): Get a particular item (including pre-processing) from the
        dataset. This function is required by `torch.utils.data.DataLoader`, and
        executes the transformation pipeline with CPU. Please make sure this
        function ALWAYS works.
    (5) define_dali_graph(): Define the data pre-processing (excluding loading
        raw data from disk) graph for DALI (Data Loading Library from NVIDIA).
        This function only works when every individual transformation node in
        the transformation pipeline supports DALI. Please refer to
        `datasets/transformations/base_transformation.py` and
        `datasets/data_loaders/dali_pipeline.py` for more details.

    The derived class should contain the following methods:

    Property functions:

    (1) num_raw_outputs: Number of raw outputs fetched from the dataset. This
        field should match the results returned by `self.get_raw_data()`.
        (requires implementation)
    (2) output_keys: Keys of each element within an item produced by the dataset
        after the entire transformation pipeline. For example, if the processed
        data looks like `{'image': [256, 256, 3], 'label', 0}`, this function
        should return `['image', 'label']`. This field should match the results
        returned by `self.transform()`. (requires implementation)

    Unit functions:

    (1) get_raw_data(): Get raw data (e.g., raw image and raw label) from disk
        according to the index. (requires implementation)
    (2) transform(): Define the transformation pipeline to pre-process the raw
        data. The transformed data will be used for training.
        (requires implementation)
    (3) parse_annotation_file(): Define how to parse item list from a given
        annotation file. (optional, can directly use the function provided by
        the base class)
    (4) info(): Collect the information of the dataset. (optional, can directly
        use the function provided by the base class)
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
                 transform_kwargs=None):
        """Initializes the dataset.

        Args:
            root_dir: Root directory (or file path) containing the dataset.
            file_format: Format the dataset is stored. Supports `dir`, `lmdb`,
                `tar`, and `zip`. (default: `zip`)
            annotation_path: A path to the annotation file. If set to `None`,
                dataset will turn to `annotation_meta` to parse item list.
                (default: None)
            annotation_meta: Name of the annotation file within `root_dir`.
                This file will be used only when parsing `annotation_path`
                fails. (default: None)
            annotation_format: Format of the annotation file. Support `json` and
                `txt`. (default: `json`)
            max_samples: Maximum samples used for the dataset. If set as a
                positive integer, samples with indices beyond this value will be
                ignored. If set as a non-positive integer, all samples will be
                used. (default: -1)
            mirror: Whether to mirror the dataset, i.e., flip each sample
                horizontally to enlarge the dataset twice. (default: False)
            transform_config: Configuration of the data transformation, i.e.,
                data pre-processing. (default: None)
        """
        # Build file reader to read data from disk.
        self.root_dir = root_dir
        if file_format is None:
            file_format = parse_file_format(root_dir)
        assert file_format is not None, 'Unparsable file format from root dir!'
        self.file_format = file_format.lower()
        self.reader = build_file_reader(self.file_format)

        # Parse item list of the dataset.
        self.annotation_path = annotation_path
        self.annotation_meta = annotation_meta
        if isinstance(annotation_format, str):
            annotation_format = annotation_format.lower()
        elif annotation_format is None:
            annotation_format = parse_ann_format(self.annotation_path)
        self.annotation_format = annotation_format
        self.items = None

        if self.annotation_format is not None and self.annotation_format not in _ALLOWED_ANNO_FORMATS:
            raise ValueError(f'Invalid annotation format: '
                             f'`{self.annotation_format}`!\n'
                             f'Formats allowed: {_ALLOWED_ANNO_FORMATS}.')
        # First option: use `annotation_path` if available.
        if annotation_path and os.path.isfile(annotation_path):
            if self.annotation_format == 'pkl':
                fp = open(annotation_path, 'rb')
            else:
                fp = open(annotation_path, 'r')
        # Second option: use `annotation_meta` if available.
        elif annotation_meta:
            fp = self.reader.open_anno_file(root_dir, annotation_meta)
        # No external annotation is provided.
        else:
            fp = None
        if fp is not None:  # Use external annotation if available.
            self.items = self.parse_annotation_file(fp)
            fp.close()
        else:  # Fallback: use image list provided by `self.reader`.
            self.items = self.reader.get_image_list(root_dir)
        assert isinstance(self.items, list) and len(self.items) > 0
        self.dataset_samples = len(self.items)
        self.num_samples = self.dataset_samples

        # Cut off the dataset if needed.
        self.max_samples = int(max_samples)
        if self.max_samples > 0:
            self.num_samples = min(self.num_samples, self.max_samples)

        # Mirror the dataset (double the item list) if needed.
        self.mirror = bool(mirror)
        if self.mirror:
            self.num_samples = self.num_samples * 2

        # Build transformations for data pre-processing.
        self.support_dali = False
        self.has_customized_function_for_dali = False
        self.transforms = dict()
        self.transform_config = transform_config
        self.transform_kwargs = transform_kwargs
        if self.transform_kwargs is not None:
            self.parse_transform_config()
        self.build_transformations()

    def __del__(self):
        """Destroys the dataset, particularly closes the file reader."""
        self.reader.close(self.root_dir)

    @property
    def name(self):
        """Returns the class name of the dataset."""
        return self.__class__.__name__

    def __len__(self):
        """Gets the total number of samples in the dataset."""
        return self.num_samples

    def fetch_file(self, filename):
        """Shortcut to reader's `fetch_file()`."""
        return self.reader.fetch_file(self.root_dir, filename)

    def build_transformations(self):
        """Builds each individual transformation in the pipeline."""
        for name, config in self.transform_config.items():
            self.transforms[name] = build_transformation(**config)
        if len(self.transforms) > 0:
            # To enable DALI pre-processing, all the nodes within the
            # transformation pipeline should support DALI.
            self.support_dali = all(trans.support_dali
                                    for trans in self.transforms.values())
            # Check if any transformation node is implemented with customized
            # function.
            self.has_customized_function_for_dali = any(
                trans.has_customized_function_for_dali
                for trans in self.transforms.values())

    def __getitem__(self, idx):
        """Gets a particular sample (with label if needed).

        NOTE: This function maps `self.output_keys` to the processed elements.

        Args:
            idx: Index of the item within the item list maintained by the
                dataset.

        Returns:
            A processed data item, which is a dictionary with `self.num_outputs`
                key-value pairs and with `self.output_keys` as the keys.
        """
        raw_data = self.get_raw_data(idx)
        transformed_data = self.transform(raw_data, use_dali=False)
        assert isinstance(transformed_data, (list, tuple))
        assert len(transformed_data) == len(self.output_keys), 'Wrong keys!'
        return dict(zip(self.output_keys, transformed_data))

    def define_dali_graph(self, raw_data):
        """Defines the graph for DALI data pre-processing.

        NOTE: This function does not map `self.output_keys` to the processed
        elements. The mapping will be achieved by the batch iterator. Please
        refer to `datasets/data_loaders/dali_batch_iterator.py` for more
        details.

        Args:
            raw_data: An operator to read raw data from the disk, which is
                treated as the starting node of the entire pipeline. In fact,
                `self.get_raw_data()` is used as the source node. Please refer
                to `datasets/data_loaders/distributed_sampler.py` and
                `datasets/data_loaders/dali_pipeline.py` for more details.

        Returns:
            A list, containing the elements of a processed data item.
        """
        assert self.support_dali, 'Some transformation does not support DALI!'
        transformed_data = self.transform(raw_data, use_dali=True)
        if not isinstance(transformed_data, (list, tuple)):
            return [transformed_data]
        return list(transformed_data)

    def get_raw_data(self, idx):
        """Gets raw data of a particular item.

        Args:
            idx: Index of the item within the item list maintained by the
                dataset.

        Returns:
            The raw data fetched by the file reader.
        """
        raise NotImplementedError('Should be implemented in derived class!')

    @property
    def num_raw_outputs(self):
        """Returns the number of raw outputs.

        This function should align with `self.get_raw_data()`, and is
        particularly used by `datasets/data_loaders/dali_pipeline.py`
        """
        raise NotImplementedError('Should be implemented in derived class!')

    def transform(self, raw_data, use_dali=False):
        """Applies data transformation for pre-processing.

        Args:
            raw_data: The raw data fetched from dick.
            use_dali: Whether to use the operations from DALI for data
                transformation. (default: False)

        Returns:
            The transformed data after pre-processing, which will be directly
                fed into the model.
        """
        raise NotImplementedError('Should be implemented in derived class!')

    @property
    def output_keys(self):
        """Returns the name of each output within a pre-processed item.

        This function should align with `self.transform()`, and is particularly
        used by `self.__getitem__()` as well as
        `datasets/data_loaders/dali_batch_iterator.py`.
        """
        raise NotImplementedError('Should be implemented in derived class!')

    def parse_annotation_file(self, fp):
        """Parses items according to the given annotation file.

        The base class provides a commonly used parsing method, which is to
        parse a JSON file directly, OR parse a TXT file by treating each line as
        a `space-joined` string.

        Please override this function in derived class if needed.

        Args:
            fp: A file pointer, pointing to the opened annotation file.

        Returns:
            A parsed item list.

        Raises:
            NotImplementedError: If `self.annotation_format` is not implemented.
        """
        if self.annotation_format == 'json':
            return json.load(fp)
        if self.annotation_format == 'pkl':
            return pickle.load(fp)

        if self.annotation_format == 'txt':
            items = []
            for line in fp:
                fields = line.rstrip().split(' ')
                if len(fields) == 1:
                    items.append(fields[0])
                else:
                    items.append(fields)
            return items

        raise NotImplementedError(f'Not implemented annotation format '
                                  f'`{self.annotation_format}`!')

    def info(self):
        """Collects the information of the dataset.

        Please append new information in derived class if needed.
        """
        dataset_info = {
            'Type': self.name,
            'Root dir': self.root_dir,
            'Dataset file format': self.file_format,
            'Annotation path': self.annotation_path,
            'Annotation meta': self.annotation_meta,
            'Annotation format': self.annotation_format,
            'Num samples in dataset': self.dataset_samples,
            'Num samples to use (negative means all)': self.max_samples,
            'Mirror': self.mirror,
            'Actual num samples used (after mirror)': self.num_samples,
            'Support DALI': self.support_dali,
            'Has customized function for DALI forwarding':
                self.has_customized_function_for_dali
        }
        return dataset_info
