# python3.7
"""Contains the VGG16 model, which is used for inference ONLY.

VGG16 is commonly used for perceptual feature extraction. The model implemented
in this file can be used for evaluation (like computing LPIPS, perceptual path
length, etc.), OR be used in training for loss computation (like perceptual
loss, etc.).

The pre-trained model is officially shared by

https://www.robots.ox.ac.uk/~vgg/research/very_deep/

and ported by

https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt

Compared to the official VGG16 model, this ported model also support evaluating
LPIPS, which is introduced in

https://github.com/richzhang/PerceptualSimilarity
"""

import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from utils.misc import download_url

__all__ = ['PerceptualModel']

# pylint: disable=line-too-long
_MODEL_URL_SHA256 = {
    # This model is provided by `torchvision`, which is ported from TensorFlow.
    'torchvision_official': (
        'https://download.pytorch.org/models/vgg16-397923af.pth',
        '397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0'  # hash sha256
    ),

    # This model is provided by https://github.com/NVlabs/stylegan2-ada-pytorch
    'vgg_perceptual_lpips': (
        'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt',
        'b437eb095feaeb0b83eb3fa11200ebca4548ee39a07fb944a417ddc516cc07c3'  # hash sha256
    )
}
# pylint: enable=line-too-long


class PerceptualModel(object):
    """Defines the perceptual model, which is based on VGG16 structure.

    This is a static class, which is used to avoid this model to be built
    repeatedly. Consequently, this model is particularly used for inference,
    like computing LPIPS, or for loss computation, like perceptual loss. If
    training is required, please use the model from `torchvision.models` or
    implement by yourself.

    NOTE: The pre-trained model assumes the inputs to be with `RGB` channel
    order and pixel range [-1, 1], and will NOT resize the input automatically
    if only perceptual feature is needed.
    """
    models = dict()

    @staticmethod
    def build_model(use_torchvision=False, no_top=True, enable_lpips=True):
        """Builds the model and load pre-trained weights.

        1. If `use_torchvision` is set as True, the model released by
           `torchvision` will be loaded, otherwise, the model released by
           https://www.robots.ox.ac.uk/~vgg/research/very_deep/ will be used.
           (default: False)

        2. To save computing resources, these is an option to only load the
           backbone (i.e., without the last three fully-connected layers). This
           is commonly used for perceptual loss or LPIPS loss computation.
           Please use argument `no_top` to control this. (default: True)

        3. For LPIPS loss computation, some additional weights (which is used
           for balancing the features from different resolutions) are employed
           on top of the original VGG16 backbone. Details can be found at
           https://github.com/richzhang/PerceptualSimilarity. Please use
           `enable_lpips` to enable this feature. (default: True)

        The built model supports following arguments when forwarding:

        - resize_input: Whether to resize the input image to size [224, 224]
            before forwarding. For feature-based computation (i.e., only
            convolutional layers are used), image resizing is not essential.
            (default: False)
        - return_tensor: This field resolves the model behavior. Following
            options are supported:
                `feature1`: Before the first max pooling layer.
                `pool1`: After the first max pooling layer.
                `feature2`: Before the second max pooling layer.
                `pool2`: After the second max pooling layer.
                `feature3`: Before the third max pooling layer.
                `pool3`: After the third max pooling layer.
                `feature4`: Before the fourth max pooling layer.
                `pool4`: After the fourth max pooling layer.
                `feature5`: Before the fifth max pooling layer.
                `pool5`: After the fifth max pooling layer.
                `flatten`: The flattened feature, after `adaptive_avgpool`.
                `feature`: The 4096d feature for logits computation. (default)
                `logits`: The 1000d categorical logits.
                `prediction`: The 1000d predicted probability.
                `lpips`: The LPIPS score between two input images.
        """
        if use_torchvision:
            model_source = 'torchvision_official'
            align_tf_resize = False
            is_torch_script = False
        else:
            model_source = 'vgg_perceptual_lpips'
            align_tf_resize = True
            is_torch_script = True

        if enable_lpips and model_source != 'vgg_perceptual_lpips':
            warnings.warn('The pre-trained model officially released by '
                          '`torchvision` does not support LPIPS computation! '
                          'Equal weights will be used for each resolution.')

        fingerprint = (model_source, no_top, enable_lpips)

        if fingerprint not in PerceptualModel.models:
            # Build model.
            model = VGG16(align_tf_resize=align_tf_resize,
                          no_top=no_top,
                          enable_lpips=enable_lpips)

            # Download pre-trained weights.
            if dist.is_initialized() and dist.get_rank() != 0:
                dist.barrier()  # Download by chief.

            url, sha256 = _MODEL_URL_SHA256[model_source]
            filename = f'perceptual_model_{model_source}_{sha256}.pth'
            model_path, hash_check = download_url(url,
                                                  filename=filename,
                                                  sha256=sha256)
            if is_torch_script:
                src_state_dict = torch.jit.load(model_path, map_location='cpu')
            else:
                src_state_dict = torch.load(model_path, map_location='cpu')
            if hash_check is False:
                warnings.warn(f'Hash check failed! The remote file from URL '
                              f'`{url}` may be changed, or the downloading is '
                              f'interrupted. The loaded perceptual model may '
                              f'have unexpected behavior.')

            if dist.is_initialized() and dist.get_rank() == 0:
                dist.barrier()  # Wait for other replicas.

            # Load weights.
            dst_state_dict = _convert_weights(src_state_dict, model_source)
            model.load_state_dict(dst_state_dict, strict=False)
            del src_state_dict, dst_state_dict

            # For inference only.
            model.eval().requires_grad_(False).cuda()
            PerceptualModel.models[fingerprint] = model

        return PerceptualModel.models[fingerprint]


def _convert_weights(src_state_dict, model_source):
    if model_source not in _MODEL_URL_SHA256:
        raise ValueError(f'Invalid model source `{model_source}`!\n'
                         f'Sources allowed: {list(_MODEL_URL_SHA256.keys())}.')
    if model_source == 'torchvision_official':
        dst_to_src_var_mapping = {
            'conv11.weight': 'features.0.weight',
            'conv11.bias': 'features.0.bias',
            'conv12.weight': 'features.2.weight',
            'conv12.bias': 'features.2.bias',
            'conv21.weight': 'features.5.weight',
            'conv21.bias': 'features.5.bias',
            'conv22.weight': 'features.7.weight',
            'conv22.bias': 'features.7.bias',
            'conv31.weight': 'features.10.weight',
            'conv31.bias': 'features.10.bias',
            'conv32.weight': 'features.12.weight',
            'conv32.bias': 'features.12.bias',
            'conv33.weight': 'features.14.weight',
            'conv33.bias': 'features.14.bias',
            'conv41.weight': 'features.17.weight',
            'conv41.bias': 'features.17.bias',
            'conv42.weight': 'features.19.weight',
            'conv42.bias': 'features.19.bias',
            'conv43.weight': 'features.21.weight',
            'conv43.bias': 'features.21.bias',
            'conv51.weight': 'features.24.weight',
            'conv51.bias': 'features.24.bias',
            'conv52.weight': 'features.26.weight',
            'conv52.bias': 'features.26.bias',
            'conv53.weight': 'features.28.weight',
            'conv53.bias': 'features.28.bias',
            'fc1.weight': 'classifier.0.weight',
            'fc1.bias': 'classifier.0.bias',
            'fc2.weight': 'classifier.3.weight',
            'fc2.bias': 'classifier.3.bias',
            'fc3.weight': 'classifier.6.weight',
            'fc3.bias': 'classifier.6.bias',
        }
    elif model_source == 'vgg_perceptual_lpips':
        src_state_dict = src_state_dict.state_dict()
        dst_to_src_var_mapping = {
            'conv11.weight': 'layers.conv1.weight',
            'conv11.bias': 'layers.conv1.bias',
            'conv12.weight': 'layers.conv2.weight',
            'conv12.bias': 'layers.conv2.bias',
            'conv21.weight': 'layers.conv3.weight',
            'conv21.bias': 'layers.conv3.bias',
            'conv22.weight': 'layers.conv4.weight',
            'conv22.bias': 'layers.conv4.bias',
            'conv31.weight': 'layers.conv5.weight',
            'conv31.bias': 'layers.conv5.bias',
            'conv32.weight': 'layers.conv6.weight',
            'conv32.bias': 'layers.conv6.bias',
            'conv33.weight': 'layers.conv7.weight',
            'conv33.bias': 'layers.conv7.bias',
            'conv41.weight': 'layers.conv8.weight',
            'conv41.bias': 'layers.conv8.bias',
            'conv42.weight': 'layers.conv9.weight',
            'conv42.bias': 'layers.conv9.bias',
            'conv43.weight': 'layers.conv10.weight',
            'conv43.bias': 'layers.conv10.bias',
            'conv51.weight': 'layers.conv11.weight',
            'conv51.bias': 'layers.conv11.bias',
            'conv52.weight': 'layers.conv12.weight',
            'conv52.bias': 'layers.conv12.bias',
            'conv53.weight': 'layers.conv13.weight',
            'conv53.bias': 'layers.conv13.bias',
            'fc1.weight': 'layers.fc1.weight',
            'fc1.bias': 'layers.fc1.bias',
            'fc2.weight': 'layers.fc2.weight',
            'fc2.bias': 'layers.fc2.bias',
            'fc3.weight': 'layers.fc3.weight',
            'fc3.bias': 'layers.fc3.bias',
            'lpips.0.weight': 'lpips0',
            'lpips.1.weight': 'lpips1',
            'lpips.2.weight': 'lpips2',
            'lpips.3.weight': 'lpips3',
            'lpips.4.weight': 'lpips4',
        }
    else:
        raise NotImplementedError(f'Not implemented model source '
                                  f'`{model_source}`!')

    dst_state_dict = {}
    for dst_name, src_name in dst_to_src_var_mapping.items():
        if dst_name.startswith('lpips'):
            dst_state_dict[dst_name] = src_state_dict[src_name].unsqueeze(0)
        else:
            dst_state_dict[dst_name] = src_state_dict[src_name].clone()
    return dst_state_dict


_IMG_MEAN = (0.485, 0.456, 0.406)
_IMG_STD  = (0.229, 0.224, 0.225)
_ALLOWED_RETURN = [
    'feature1', 'pool1', 'feature2', 'pool2', 'feature3', 'pool3', 'feature4',
    'pool4', 'feature5', 'pool5', 'flatten', 'feature', 'logits', 'prediction',
    'lpips'
]

# pylint: disable=missing-function-docstring

class VGG16(nn.Module):
    """Defines the VGG16 structure.

    This model takes `RGB` images with data format `NCHW` as the raw inputs. The
    pixel range are assumed to be [-1, 1].
    """

    def __init__(self, align_tf_resize=False, no_top=True, enable_lpips=True):
        """Defines the network structure."""
        super().__init__()

        self.align_tf_resize = align_tf_resize
        self.no_top = no_top
        self.enable_lpips = enable_lpips

        self.conv11 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu11 = nn.ReLU(inplace=True)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu12 = nn.ReLU(inplace=True)
        # output `feature1`, with shape [N, 64, 224, 224]

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # output `pool1`, with shape [N, 64, 112, 112]

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu21 = nn.ReLU(inplace=True)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu22 = nn.ReLU(inplace=True)
        # output `feature2`, with shape [N, 128, 112, 112]

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # output `pool2`, with shape [N, 128, 56, 56]

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu31 = nn.ReLU(inplace=True)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu32 = nn.ReLU(inplace=True)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu33 = nn.ReLU(inplace=True)
        # output `feature3`, with shape [N, 256, 56, 56]

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # output `pool3`, with shape [N,256, 28, 28]

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.relu41 = nn.ReLU(inplace=True)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu42 = nn.ReLU(inplace=True)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu43 = nn.ReLU(inplace=True)
        # output `feature4`, with shape [N, 512, 28, 28]

        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # output `pool4`, with shape [N, 512, 14, 14]

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu51 = nn.ReLU(inplace=True)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu52 = nn.ReLU(inplace=True)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu53 = nn.ReLU(inplace=True)
        # output `feature5`, with shape [N, 512, 14, 14]

        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        # output `pool5`, with shape [N, 512, 7, 7]

        if self.enable_lpips:
            self.lpips = nn.ModuleList()
            for idx, ch in enumerate([64, 128, 256, 512, 512]):
                self.lpips.append(nn.Conv2d(ch, 1, kernel_size=1, bias=False))
                self.lpips[idx].weight.data.copy_(torch.ones(1, ch, 1, 1))

        if not self.no_top:
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
            # output `flatten`, with shape [N, 25088]

            self.fc1 = nn.Linear(512 * 7 * 7, 4096)
            self.fc1_relu = nn.ReLU(inplace=True)
            self.fc1_dropout = nn.Dropout(0.5, inplace=False)
            self.fc2 = nn.Linear(4096, 4096)
            self.fc2_relu = nn.ReLU(inplace=True)
            self.fc2_dropout = nn.Dropout(0.5, inplace=False)
            # output `feature`, with shape [N, 4096]

            self.fc3 = nn.Linear(4096, 1000)
            # output `logits`, with shape [N, 1000]

            self.out = nn.Softmax(dim=1)
            # output `softmax`, with shape [N, 1000]

        img_mean = np.array(_IMG_MEAN).reshape((1, 3, 1, 1)).astype(np.float32)
        img_std = np.array(_IMG_STD).reshape((1, 3, 1, 1)).astype(np.float32)
        self.register_buffer('img_mean', torch.from_numpy(img_mean))
        self.register_buffer('img_std', torch.from_numpy(img_std))

    def forward(self,
                x,
                y=None,
                *,
                resize_input=False,
                return_tensor='feature'):
        return_tensor = return_tensor.lower()
        if return_tensor not in _ALLOWED_RETURN:
            raise ValueError(f'Invalid output tensor name `{return_tensor}` '
                             f'for perceptual model (VGG16)!\n'
                             f'Names allowed: {_ALLOWED_RETURN}.')

        if return_tensor == 'lpips' and y is None:
            raise ValueError('Two images are required for LPIPS computation, '
                             'but only one is received!')

        if return_tensor == 'lpips':
            assert x.shape == y.shape
            x = torch.cat([x, y], dim=0)
            features = []

        if resize_input:
            if self.align_tf_resize:
                theta = torch.eye(2, 3).to(x)
                theta[0, 2] += theta[0, 0] / x.shape[3] - theta[0, 0] / 224
                theta[1, 2] += theta[1, 1] / x.shape[2] - theta[1, 1] / 224
                theta = theta.unsqueeze(0).repeat(x.shape[0], 1, 1)
                grid = F.affine_grid(theta,
                                     size=(x.shape[0], x.shape[1], 224, 224),
                                     align_corners=False)
                x = F.grid_sample(x, grid,
                                  mode='bilinear',
                                  padding_mode='border',
                                  align_corners=False)
            else:
                x = F.interpolate(x,
                                  size=(224, 224),
                                  mode='bilinear',
                                  align_corners=False)
        if x.shape[1] == 1:
            x = x.repeat((1, 3, 1, 1))

        x = (x + 1) / 2
        x = (x - self.img_mean) / self.img_std

        x = self.conv11(x)
        x = self.relu11(x)
        x = self.conv12(x)
        x = self.relu12(x)
        if return_tensor == 'feature1':
            return x
        if return_tensor == 'lpips':
            features.append(x)

        x = self.pool1(x)
        if return_tensor == 'pool1':
            return x

        x = self.conv21(x)
        x = self.relu21(x)
        x = self.conv22(x)
        x = self.relu22(x)
        if return_tensor == 'feature2':
            return x
        if return_tensor == 'lpips':
            features.append(x)

        x = self.pool2(x)
        if return_tensor == 'pool2':
            return x

        x = self.conv31(x)
        x = self.relu31(x)
        x = self.conv32(x)
        x = self.relu32(x)
        x = self.conv33(x)
        x = self.relu33(x)
        if return_tensor == 'feature3':
            return x
        if return_tensor == 'lpips':
            features.append(x)

        x = self.pool3(x)
        if return_tensor == 'pool3':
            return x

        x = self.conv41(x)
        x = self.relu41(x)
        x = self.conv42(x)
        x = self.relu42(x)
        x = self.conv43(x)
        x = self.relu43(x)
        if return_tensor == 'feature4':
            return x
        if return_tensor == 'lpips':
            features.append(x)

        x = self.pool4(x)
        if return_tensor == 'pool4':
            return x

        x = self.conv51(x)
        x = self.relu51(x)
        x = self.conv52(x)
        x = self.relu52(x)
        x = self.conv53(x)
        x = self.relu53(x)
        if return_tensor == 'feature5':
            return x
        if return_tensor == 'lpips':
            features.append(x)

        x = self.pool5(x)
        if return_tensor == 'pool5':
            return x

        if return_tensor == 'lpips':
            score = 0
            assert len(features) == 5
            for idx in range(5):
                feature = features[idx]
                norm = feature.norm(dim=1, keepdim=True)
                feature = feature / (norm + 1e-10)
                feature_x, feature_y = feature.chunk(2, dim=0)
                diff = (feature_x - feature_y).square()
                score += self.lpips[idx](diff).mean(dim=(2, 3), keepdim=False)
            return score.sum(dim=1, keepdim=False)

        x = self.avgpool(x)
        x = self.flatten(x)
        if return_tensor == 'flatten':
            return x

        x = self.fc1(x)
        x = self.fc1_relu(x)
        x = self.fc1_dropout(x)
        x = self.fc2(x)
        x = self.fc2_relu(x)
        x = self.fc2_dropout(x)
        if return_tensor == 'feature':
            return x

        x = self.fc3(x)
        if return_tensor == 'logits':
            return x

        x = self.out(x)
        if return_tensor == 'prediction':
            return x

        raise NotImplementedError(f'Output tensor name `{return_tensor}` is '
                                  f'not implemented!')

# pylint: enable=missing-function-docstring
