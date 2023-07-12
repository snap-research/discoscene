# python3.7
"""Contains the class of image dataset.

`ImageDataset` is commonly used as the dataset that provides images with labels.
Concretely, each data sample (or say item) consists of an image and its
corresponding label (if provided).
"""

import numpy as np
import random


from utils.formatting_utils import raw_label_to_one_hot
from .base_dataset import BaseDataset
from .transformations import switch_between
import cv2

__all__ = ['ImageBboxDataset']


class ImageBboxDataset(BaseDataset):
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
                 num_classes=None,
                 # cam_path=None,
                 # sample_rotation=False,
                 use_bbox_2d=False,
                 # use_object=False,
                 # use_mask=False,
                 # use_bbox_label=False,
                 num_bbox=None,
                 enable_flip=False):
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
        self.num_classes = num_classes  # Actual number of classes provided by the loader.

        # Check if the dataset contains categorical information.
        self.use_label = False
        self.num_bbox = num_bbox
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
            self.num_classes = num_classes

        # self.cam_matrix = None
        # if cam_path is not None:
        #     try:
        #         self.cam_matrix = np.load(cam_path, allow_pickle=True).item() 
        #     except:
        #         self.cam_matrix = None

        # self.use_object = use_object
        # self.sample_rotation = sample_rotation
        self.use_bbox_2d = use_bbox_2d
        # self.use_mask = use_mask
        # self.use_bbox_label = use_bbox_label
        self.enable_flip = enable_flip
        # if self.use_object: assert self.cam_matrix is not None

    def project(self, bbox, cam_matrix, scale):
        RT = np.array(cam_matrix['RT'])
        K = np.array(cam_matrix['K'])
        R = RT[:, :3]
        T = RT[:, 3:]
        
        xyz = bbox
        xyz = np.dot(xyz, R.T) + T.T
        xyz = np.dot(xyz, K.T)
        xy = xyz[..., :2] / xyz[..., 2:]
        # xy = (xy/scale).astype(np.uint8)
        return xy

    def get_bbox(self, idx):
        bbox_item = self.items[idx]
        seed = np.random.randint(10e7) 
        g_bbox = self.pad(np.array(bbox_item['bbox']), seed=seed)
        g_cano_bbox = self.pad(np.array(bbox_item['cano_bbox']), seed=seed)
        g_bbox_scale = self.pad(np.array(bbox_item['bbox_scale']), seed=seed)
        g_bbox_rot = self.pad(np.array(bbox_item['bbox_rot']), seed=seed)
        g_bbox_tran = self.pad(np.array(bbox_item['bbox_tran']), seed=seed)
        try:# specify the validation data
            g_image_raw_scale = bbox_item['size'][0]
            g_image = np.ones((g_image_raw_scale, g_image_raw_scale, 3), dtype=np.uint8)
        except: # specify the training data
            buffer = np.frombuffer(self.fetch_file(bbox_item['image_path']), dtype=np.uint8) 
            g_image = self.transforms['decode'](buffer, use_dali=False) 
            g_image = g_image[:,:,:3]
            g_image_raw_scale = g_image.shape[0]
        # g_image = np.ones((1920,1920,3), dtype=np.uint8)
        # g_image = np.ones((512, 512,3), dtype=np.uint8)
        # g_image = np.ones((256, 256,3), dtype=np.uint8)
        g_bbox_valid = self.pad(np.ones(len(bbox_item['bbox'])), seed=seed)
        bbox_results = {'g_bbox': g_bbox,
                        'g_cano_bbox': g_cano_bbox,
                        'g_bbox_scale': g_bbox_scale,
                        'g_bbox_rot': g_bbox_rot,
                        'g_bbox_tran': g_bbox_tran,
                        'g_image': g_image,
                        'g_image_raw_scale': g_image_raw_scale,
                        'g_bbox_valid': g_bbox_valid}

        assert 'RT' in bbox_item and 'K' in bbox_item
        cam_matrix = dict(RT=bbox_item['RT'],
                          K=bbox_item['K'])
        bbox_results.update(g_bbox_RT=np.array(bbox_item['RT']),
                            g_bbox_K=np.array(bbox_item['K']))
        g_img_bbox = self.project(g_bbox, cam_matrix, 4)        

        return bbox_results

    def pad(self, array, seed):
        if self.num_bbox is None or self.num_bbox == array.shape[0]:
            return array
        elif self.num_bbox < array.shape[0]:
            random.seed(seed)
            np.random.seed(seed)
            index = list(range(array.shape[0]))
            random.shuffle(index)
            s_index = index[:self.num_bbox]
            # print(seed, s_index)
            return array[s_index]
        else:
            pad_array = np.ones((self.num_bbox, ) + array.shape[1:]) * -1.0
            pad_array[:array.shape[0]] = array
            return pad_array

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
            item = self.items[idx]
            bbox_idx = np.random.randint(len(self.items))
            bbox_item = self.items[bbox_idx]
            if isinstance(item, dict):
                image_path = item['image_path'] 
                
                seed = np.random.randint(10e7) 
                bbox = self.pad(np.array(item['bbox']), seed=seed)
                bbox_scale = self.pad(np.array(item['bbox_scale']), seed=seed)
                bbox_rot = self.pad(np.array(item['bbox_rot']), seed=seed)
                bbox_tran = self.pad(np.array(item['bbox_tran']), seed=seed)
                bbox_valid = self.pad(np.ones(len(item['bbox'])), seed=seed)
                #TODO scale
                assert 'RT' in item and 'K' in item 
                cam_matrix = dict(RT=item['RT'],
                                  K=item['K'])
                item['K'] = np.array(item['K']).astype(np.float32)
                item['RT'] = np.array(item['RT']).astype(np.float32)
                if self.use_bbox_2d:
                    image_bbox = self.pad(np.array(item['bbox_image']), seed=seed)
                else:
                    image_bbox = self.pad(self.project(bbox, cam_matrix, 4), seed=seed)       

                seed = np.random.randint(10e7) 
                g_bbox = self.pad(np.array(bbox_item['bbox']), seed=seed)
                g_cano_bbox = self.pad(np.array(bbox_item['cano_bbox']), seed=seed)
                g_bbox_scale = self.pad(np.array(bbox_item['bbox_scale']), seed=seed)
                g_bbox_rot = self.pad(np.array(bbox_item['bbox_rot']), seed=seed)
                g_bbox_tran = self.pad(np.array(bbox_item['bbox_tran']), seed=seed)
                g_bbox_valid = self.pad(np.ones(len(bbox_item['bbox'])), seed=seed)

                assert 'RT' in item and 'K' in bbox_item 
                cam_matrix = dict(RT=bbox_item['RT'],
                                  K=bbox_item['K'])
                bbox_item['K'] = np.array(bbox_item['K']).astype(np.float32)
                bbox_item['RT'] = np.array(bbox_item['RT']).astype(np.float32)
                g_image_bbox = self.pad(self.project(g_bbox, cam_matrix, 4), seed=seed)   
            else:
                image_path = self.items[idx]

        # Load image to buffer.
        buffer = np.frombuffer(self.fetch_file(image_path), dtype=np.uint8)

        idx = np.array(idx)
        do_mirror = np.array(do_mirror)
        if self.use_label:
            raw_label = np.array(raw_label)
            return [idx, do_mirror, buffer, raw_label, label]
        return [idx, do_mirror, buffer, bbox, bbox_scale, bbox_rot, bbox_tran, image_bbox, bbox_valid, np.array(item['K']),  np.array(item['RT']), \
                g_bbox, g_cano_bbox, g_bbox_scale, g_bbox_rot, g_bbox_tran, g_image_bbox, g_bbox_valid, np.array(bbox_item['K']),  np.array(bbox_item['RT'])]

    @property
    def num_raw_outputs(self):
        if self.use_label:
            return 5  # [idx, do_mirror, buffer, raw_label, label]
        if self.use_object:
            return 3 + 6 +2
        return 3+6  # [idx, do_mirror, buffer]

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
            idx, do_mirror, buffer, bbox, bbox_scale, bbox_rot, bbox_tran, image_bbox, bbox_valid, bbox_K, bbox_RT,  g_bbox, g_cano_bbox, g_bbox_scale, g_bbox_rot, g_bbox_tran, g_image_bbox, g_bbox_valid, g_bbox_K, g_bbox_RT = raw_data


        # TODO need to be check
        op_list = [key for key in self.transforms.keys() if key not in ['decode', 'horizontal_flip', 'normalize']]
        raw_image = self.transforms['decode'](buffer, use_dali=use_dali)
        raw_image = raw_image[:,:,:3]
        H, W = raw_image.shape[:2]

        if self.enable_flip:
            is_real_flip = False
            for idx in range(len(bbox_valid)):
                if bbox_valid[idx] == 1:
                    angle = cv2.Rodrigues(bbox_rot[idx])[0][1]
                    dist = min(np.abs(angle - np.pi/2), np.abs(angle+np.pi/2))
                    if dist >= (np.pi/2)*2.5/4: 
                        # print('angle', angle)
                        is_real_flip = True
                        break
            if is_real_flip:
                do_real_flip = np.random.uniform() < 0.5
                if do_real_flip:
                    # print('do real flip')
                    image_bbox[:, :, 0] =  W - 1 - image_bbox[:, :, 0]

            
            is_fake_flip = False
            for idx in range(len(g_bbox_valid)):
                if g_bbox_valid[idx] == 1:
                    angle = cv2.Rodrigues(g_bbox_rot[idx])[0][1]
                    dist = min(np.abs(angle - np.pi/2), np.abs(angle+np.pi/2))
                    if dist >= (np.pi/2)*3/4: 
                        is_fake_flip = True
                        break

            if is_fake_flip:
                do_fake_flip = np.random.uniform() < 0.5
                if do_fake_flip:
                    for idx in range(len(g_bbox_valid)):
                        if g_bbox_valid[idx] == 1:
                            g_centers = (np.linalg.inv(g_bbox_K) @ np.array([(W-1)/2, (W-1)/2, 1])*g_bbox[idx][:, 2:])
                            g_image_bbox[idx][:, 0] = W - 1 - g_image_bbox[idx][:, 0]
                            g_bbox[idx][:, 0] = 2*g_centers[:, 0] - g_bbox[idx][:, 0]
                            g_bbox_tran[idx][0,0] = 2*g_centers[:, 0].mean(axis=0) - g_bbox_tran[idx][0, 0]
                            angle = cv2.Rodrigues(g_bbox_rot[idx])[0]
                            angle[1, 0] = (-np.pi/2 + -0.0014218876641862463)*2 - angle[1, 0]
                            g_bbox_rot[idx] = cv2.Rodrigues(angle)[0]
                            cbbox = (g_bbox_rot[idx].T @ (g_bbox[idx].T - g_bbox_tran[idx].reshape(3, -1)))/(g_bbox_scale[idx].reshape(3,-1))
                            # print(cbbox)

        # TODO we assume the scale of g_image is the same as the image
        g_image_raw_scale = raw_image.shape[0]
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
        if self.enable_flip:
            if is_real_flip:
                if do_real_flip:
                    # print('flip real image')
                    image = image[:, ::-1]

        image = self.transforms['normalize'](image, use_dali=use_dali)
        image_scale = image.shape[-1]

        if self.use_label:
            return [idx, raw_image, image, raw_label, label]
        else:
            return [idx, raw_image, image, bbox, bbox_scale, bbox_rot, bbox_tran, image_bbox, bbox_valid, bbox_K, bbox_RT,  g_bbox, g_cano_bbox, g_bbox_scale, g_bbox_rot, g_bbox_tran, g_image_bbox, g_bbox_valid, g_bbox_K, g_bbox_RT, g_image_raw_scale] 

    @property
    def output_keys(self):
        if self.use_label:
            return ['index', 'raw_image', 'image', 'raw_label', 'label']
        return ['index', 'raw_image', 'image', 'bbox', 'bbox_scale', 'bbox_rot', 'bbox_tran', 'image_bbox', 'bbox_valid', 'bbox_K', 'bbox_RT', 'g_bbox', 'g_cano_bbox', 'g_bbox_scale', 'g_bbox_rot', 'g_bbox_tran', 'g_image_bbox', 'g_bbox_valid', 'g_bbox_K', 'g_bbox_RT', 'g_image_raw_scale']
            

    def __getitem__(self, idx):
        raw_data = self.get_raw_data(idx)
        transformed_data = self.transform(raw_data, use_dali=False)
        assert isinstance(transformed_data, (list, tuple))
        assert len(transformed_data) == len(self.output_keys), 'Wrong keys!'
        return dict(zip(self.output_keys, transformed_data))

    def info(self):
        dataset_info = super().info()
        dataset_info['Dataset classes'] = self.dataset_classes
        dataset_info['Use label'] = self.use_label
        if self.use_label:
            dataset_info['Num classes for training'] = self.num_classes
        return dataset_info
