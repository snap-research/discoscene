# python3.7
"""Script to prepare dataset in `zip` format."""

import os
import io
import warnings
import argparse
import json
import zipfile
from tqdm import tqdm
from packaging import version

import torchvision
import torchvision.datasets as torch_datasets

_ALLOWED_DATASETS = [
    'folder', 'cifar10', 'cifar100', 'mnist', 'imagenet1k', 'lsun',
    'inaturalist'
]


def adapt_stylegan2ada_dataset(dataset_path, annotation_meta='annotation.json'):
    """Adapts a dataset created by official StyleGAN2-ADA.

    Please refer to the link below for more details:

    https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/dataset_tool.py

    Concretely, this function parses the `dataset.json` inside the original
    dataset, then de-wraps the `labels` key, and finally adds a new annotation
    file, `annotation_meta`, into the dataset with JSON format.

    Args:
        dataset_path: `str`, path to the original dataset.
        annotation_meta: `str`, name of the new annotation file saved in the
            dataset. (default: `annotation.json`)
    """
    # File will be closed after the function execution.
    zip_file = zipfile.ZipFile(dataset_path, 'a')  # pylint: disable=consider-using-with

    # Early return if no annotation file is found.
    if 'dataset.json' not in zip_file.namelist():
        zip_file.close()
        return

    # Parse annotation from the source file.
    with zip_file.open('dataset.json', 'r') as f:
        dataset_annotations = json.load(f)
        dataset_annotations = dataset_annotations.get('labels', None)
    # Add the new annotation file with JSON format.
    zip_file.writestr(annotation_meta,
                      data=json.dumps(dataset_annotations))

    zip_file.close()


def open_dataset(path, dataset='folder', portion='train'):
    """Opens a dataset with specified portion (if available).

    Args:
        path: `str`, path/directory to the raw data.
        dataset: `str`, name of the dataset. (default: `folder`)
        portion: `str`, portion of dataset to be used. This field may be ignored
            if the portion is not available (e.g., when `dataset` is `folder`).
            (default: `train`)

    Returns:
        A `torch.utils.data.Dataset` that iterates over data.

    Raises:
        ValueError: If the input `dataset` is not supported.
        NotImplementedError: If the input `dataset` is not implemented.
    """
    dataset = dataset.lower()
    portion = portion.lower()

    if dataset not in _ALLOWED_DATASETS:
        raise ValueError(f'Invalid dataset: `{dataset}`!\n'
                         f'Supported datasets: {_ALLOWED_DATASETS}.')

    # Image Folder
    if dataset == 'folder':
        assert os.path.isdir(path)
        return torch_datasets.ImageFolder(path)

    # CIFAR-10
    if dataset == 'cifar10':
        data_exist = os.path.isfile(path)
        return torch_datasets.CIFAR10(path,
                                      train=(portion == 'train'),
                                      download=not data_exist)

    # CIFAR-100
    if dataset == 'cifar100':
        data_exist = os.path.isfile(path)
        return torch_datasets.CIFAR100(path,
                                       train=(portion == 'train'),
                                       download=not data_exist)

    # MNIST
    if dataset == 'mnist':
        data_exist = os.path.isfile(path)
        return torch_datasets.MNIST(path,
                                    train=(portion == 'train'),
                                    download=not data_exist)

    # ImageNet 1k / ILSVRC 2012
    if dataset  == 'imagenet1k':
        warnings.warn('It may take some time to extract raw data ...')
        return torch_datasets.ImageNet(path, split=portion)

    # LSUN
    if dataset == 'lsun':
        # For LSUN, multi-portions are supported via concatenating portion names
        # with comma. For example, passing 'bridge_train,kitchen_train' to zip
        # 'bridge_train' and 'kitchen_train' together.
        portion = list(portion.replace(' ', '').split(','))
        return torch_datasets.LSUN(path, classes=portion)

    # iNaturalist
    if dataset == 'inaturalist':
        if version.parse(torchvision.__version__) < version.parse('0.11'):
            raise ValueError('iNaturalist is not supported in your current '
                             'environment, please upgrade your `torchvision` '
                             'to 0.11 or later!')
        data_exist = os.path.isfile(os.path.join(path, f'{portion}.tgz'))
        return torch_datasets.INaturalist(path,
                                          version=portion,
                                          download=not data_exist)

    raise NotImplementedError(f'Not implemented dataset: `{dataset}`!')


def parse_meta(dataset_obj):
    """Parses the meta information about the dataset.

    Args:
        dataset_obj: a `torch.utils.data.Dataset` object, returned by function
            `open_dataset()`.

    Returns:
        A `dict` that contains the meta information.
    """
    # CIFAR10 / CIFAR 100 / MNIST
    if isinstance(dataset_obj, (torch_datasets.CIFAR10,
                                torch_datasets.CIFAR100,
                                torch_datasets.MNIST)):
        warnings.warn(f'No meta found from `{dataset_obj}`, hence skip!')
        return None

    # ImageNet 1k / ILSVRC 2012
    if isinstance(dataset_obj, torch_datasets.ImageNet):
        meta = dict(class_to_idx=dataset_obj.class_to_idx,
                    wnid_to_idx=dataset_obj.wnid_to_idx,
                    imgs=[(os.path.basename(img_path), class_idx)
                          for (img_path, class_idx) in dataset_obj.imgs])
        return meta

    # LSUN
    if isinstance(dataset_obj, torch_datasets.LSUN):
        meta = dict()
        for lsun_class in dataset_obj.dbs:
            dict_key = os.path.basename(lsun_class.root.strip('_lmdb'))
            meta[dict_key] = [key.decode('utf-8') for key in lsun_class.keys]
        return meta

    # iNaturalist
    # For compatibility in early version of torchvision.
    if version.parse(torchvision.__version__) >= version.parse('0.11'):
        if isinstance(dataset_obj, torch_datasets.INaturalist):
            meta = dict(all_categories=dataset_obj.all_categories,
                        categories_index=dataset_obj.categories_index,
                        categories_map=dataset_obj.categories_map,
                        target_type=dataset_obj.target_type,
                        index=dataset_obj.index)
            return meta

    # Folder
    # NOTE: This should be parse at last since other datasets may derive from
    # `torch_datasets.ImageFolder`
    if isinstance(dataset_obj, torch_datasets.ImageFolder):
        meta = dict(raw_imgs=dataset_obj.samples)  # (raw_path, label)
        return meta

    return None


def save_dataset(src, save_path, dataset, portion):
    """Makes and saves a dataset in `zip` format.

    Args:
        src: `str`, directory to the raw data.
        save_path: `str`, filename of the processed zipfile.
        dataset: `str`, name of the dataset.
        portion: `str`, the portion of dataset to be used.
    """
    # Open the source dataset, parse items and annotation.
    data = open_dataset(path=src, dataset=dataset, portion=portion)
    labels = []

    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=False)
    # File will be closed after the function execution.
    zip_file = zipfile.ZipFile(save_path, 'w')  # pylint: disable=consider-using-with

    # TODO: parallelize the following iteration.
    progress_bar = tqdm(enumerate(data), total=len(data), desc='Data')
    for idx, (img, target) in progress_bar:
        img_relative_path = f'{target}/img{idx:08d}.png'
        byte_img = io.BytesIO()
        img.save(byte_img, format='png', compress_level=0, optimize=False)
        zip_file.writestr(img_relative_path, byte_img.getbuffer())
        labels.append([img_relative_path, target])
    zip_file.writestr('annotation.json', data=json.dumps(labels))

    # Save meta info if exists.
    meta_info = parse_meta(data)
    if meta_info:
        zip_file.writestr('meta.json', data=json.dumps(meta_info))

    zip_file.close()


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='Prepare a `zip` dataset.')
    parser.add_argument('src', type=str,
                        help='Path to the input dataset, can be a filename or '
                             'a directory name. If the given path does not '
                             'exist, for downloadable datasets, the data will '
                             'be automatically downloaded to `src`; otherwise, '
                             'a FileNotFoundError will be raised.')
    parser.add_argument('save_path', type=str,
                        help='Path to save the dataset.')
    parser.add_argument('--dataset', type=str, default='folder',
                        choices=_ALLOWED_DATASETS,
                        help='The dataset to be used. (default: %(default)s)')
    parser.add_argument('--portion', type=str, default='train',
                        help='The portion of dataset to be used, can be '
                             '`train` or `test` split, etc., dependent on '
                             'the dataset. (default: %(default)s)')
    parser.add_argument('--anno_meta', type=str, default='annotation.json',
                        help='Name of the annotation file saved in dataset. '
                             'This field only takes effect when adapting a '
                             'dataset created by other projects (i.e., '
                             'StyleGAN2-ADA with `--dataset stylegan2ada`). '
                             '(default: %(default)s)')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Case 1: Adapt dataset from StyleGAN2-ADA.
    if args.dataset.lower() == 'stylegan2ada':
        adapt_stylegan2ada_dataset(dataset_path=args.src,
                                   annotation_meta=args.anno_meta)
        return

    # Case 2: Prepare dataset from scratch.
    save_dataset(src=args.src,
                 save_path=args.save_path,
                 dataset=args.dataset,
                 portion=args.portion)


if __name__ == '__main__':
    main()
