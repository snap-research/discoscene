# python3.7
"""Contains the Inception V3 model, which is used for inference ONLY.

This file is mostly borrowed from `torchvision/models/inception.py`.

Inception model is widely used to compute FID or IS metric for evaluating
generative models. However, the pre-trained models from torchvision is slightly
different from the TensorFlow version

http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz

which is used by the official FID implementation

https://github.com/bioinf-jku/TTUR

In particular:

(1) The number of classes in TensorFlow model is 1008 instead of 1000.
(2) The avg_pool() layers in TensorFlow model does not include the padded zero.
(3) The last Inception E Block in TensorFlow model use max_pool() instead of
    avg_pool().

Hence, to align the evaluation results with those from TensorFlow
implementation, we modified the inception model to support both versions. Please
use `align_tf` argument to control the version.
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from utils.misc import download_url

__all__ = ['InceptionModel']

# pylint: disable=line-too-long

_MODEL_URL_SHA256 = {
    # This model is provided by `torchvision`, which is ported from TensorFlow.
    'torchvision_official': (
        'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
        '1a9a5a14f40645a370184bd54f4e8e631351e71399112b43ad0294a79da290c8'  # hash sha256
    ),

    # This model is provided by https://github.com/mseitzer/pytorch-fid
    'tf_inception_v3': (
        'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth',
        '6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2'  # hash sha256
    )
}


class InceptionModel(object):
    """Defines the Inception (V3) model.

    This is a static class, which is used to avoid this model to be built
    repeatedly. Consequently, this model is particularly used for inference,
    like computing FID. If training is required, please use the model from
    `torchvision.models` or implement by yourself.

    NOTE: The pre-trained model assumes the inputs to be with `RGB` channel
    order and pixel range [-1, 1], and will also resize the images to shape
    [299, 299] automatically. If your input is normalized by subtracting
    (0.485, 0.456, 0.406) and dividing (0.229, 0.224, 0.225), please use
    `transform_input` in the `forward()` function to un-normalize it.
    """
    models = dict()

    @staticmethod
    def build_model(align_tf=True):
        """Builds the model and load pre-trained weights.

        If `align_tf` is set as True, the model will predict 1008 classes, and
        the pre-trained weight from `https://github.com/mseitzer/pytorch-fid`
        will be loaded. Otherwise, the model will predict 1000 classes, and will
        load the model from `torchvision`.

        The built model supports following arguments when forwarding:

        - transform_input: Whether to transform the input back to pixel range
            (-1, 1). Please disable this argument if your input is already with
            pixel range (-1, 1). (default: False)
        - output_logits: Whether to output the categorical logits instead of
            features. (default: False)
        - remove_logits_bias: Whether to remove the bias when computing the
            logits. The official implementation removes the bias by default.
            Please refer to
            `https://github.com/openai/improved-gan/blob/master/inception_score/model.py`.
            (default: False)
        - output_predictions: Whether to output the final predictions, i.e.,
            `softmax(logits)`. (default: False)
        """
        if align_tf:
            num_classes = 1008
            model_source = 'tf_inception_v3'
        else:
            num_classes = 1000
            model_source = 'torchvision_official'

        fingerprint = model_source

        if fingerprint not in InceptionModel.models:
            # Build model.
            model = Inception3(num_classes=num_classes,
                               aux_logits=False,
                               init_weights=False,
                               align_tf=align_tf)

            # Download pre-trained weights.
            if dist.is_initialized() and dist.get_rank() != 0:
                dist.barrier()  # Download by chief.

            url, sha256 = _MODEL_URL_SHA256[model_source]
            filename = f'inception_model_{model_source}_{sha256}.pth'
            model_path, hash_check = download_url(url,
                                                  filename=filename,
                                                  sha256=sha256)
            state_dict = torch.load(model_path, map_location='cpu')
            if hash_check is False:
                warnings.warn(f'Hash check failed! The remote file from URL '
                              f'`{url}` may be changed, or the downloading is '
                              f'interrupted. The loaded inception model may '
                              f'have unexpected behavior.')

            if dist.is_initialized() and dist.get_rank() == 0:
                dist.barrier()  # Wait for other replicas.

            # Load weights.
            model.load_state_dict(state_dict, strict=False)
            del state_dict

            # For inference only.
            model.eval().requires_grad_(False).cuda()
            InceptionModel.models[fingerprint] = model

        return InceptionModel.models[fingerprint]

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=super-with-arguments
# pylint: disable=consider-merging-isinstance
# pylint: disable=import-outside-toplevel
# pylint: disable=no-else-return

class Inception3(nn.Module):

    def __init__(self, num_classes=1000, aux_logits=True, inception_blocks=None,
                 init_weights=True, align_tf=True):
        super(Inception3, self).__init__()
        if inception_blocks is None:
            inception_blocks = [
                BasicConv2d, InceptionA, InceptionB, InceptionC,
                InceptionD, InceptionE, InceptionAux
            ]
        assert len(inception_blocks) == 7
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]

        self.aux_logits = aux_logits
        self.align_tf = align_tf
        self.Conv2d_1a_3x3 = conv_block(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        self.Mixed_5b = inception_a(192, pool_features=32, align_tf=self.align_tf)
        self.Mixed_5c = inception_a(256, pool_features=64, align_tf=self.align_tf)
        self.Mixed_5d = inception_a(288, pool_features=64, align_tf=self.align_tf)
        self.Mixed_6a = inception_b(288)
        self.Mixed_6b = inception_c(768, channels_7x7=128, align_tf=self.align_tf)
        self.Mixed_6c = inception_c(768, channels_7x7=160, align_tf=self.align_tf)
        self.Mixed_6d = inception_c(768, channels_7x7=160, align_tf=self.align_tf)
        self.Mixed_6e = inception_c(768, channels_7x7=192, align_tf=self.align_tf)
        if aux_logits:
            self.AuxLogits = inception_aux(768, num_classes)
        self.Mixed_7a = inception_d(768)
        self.Mixed_7b = inception_e(1280, align_tf=self.align_tf)
        self.Mixed_7c = inception_e(2048, use_max_pool=self.align_tf)
        self.fc = nn.Linear(2048, num_classes)
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    import scipy.stats as stats
                    stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                    X = stats.truncnorm(-2, 2, scale=stddev)
                    values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                    values = values.view(m.weight.size())
                    with torch.no_grad():
                        m.weight.copy_(values)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    @staticmethod
    def _transform_input(x, transform_input=False):
        if transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self,
                 x,
                 output_logits=False,
                 remove_logits_bias=False,
                 output_predictions=False):
        # Upsample if necessary.
        if x.shape[2] != 299 or x.shape[3] != 299:
            if self.align_tf:
                theta = torch.eye(2, 3).to(x)
                theta[0, 2] += theta[0, 0] / x.shape[3] - theta[0, 0] / 299
                theta[1, 2] += theta[1, 1] / x.shape[2] - theta[1, 1] / 299
                theta = theta.unsqueeze(0).repeat(x.shape[0], 1, 1)
                grid = F.affine_grid(theta,
                                     size=(x.shape[0], x.shape[1], 299, 299),
                                     align_corners=False)
                x = F.grid_sample(x, grid,
                                  mode='bilinear',
                                  padding_mode='border',
                                  align_corners=False)
            else:
                x = F.interpolate(
                    x, size=(299, 299), mode='bilinear', align_corners=False)
        if x.shape[1] == 1:
            x = x.repeat((1, 3, 1, 1))

        if self.align_tf:
            x = (x * 127.5 + 127.5 - 128) / 128

        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        else:
            aux = None
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        x = F.dropout(x, training=self.training)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        if output_logits or output_predictions:
            x = self.fc(x)
            # N x 1000 (num_classes)
            if remove_logits_bias:
                x = x - self.fc.bias.view(1, -1)
            if output_predictions:
                x = F.softmax(x, dim=1)
        return x, aux

    def forward(self,
                x,
                transform_input=False,
                output_logits=False,
                remove_logits_bias=False,
                output_predictions=False):
        x = self._transform_input(x, transform_input)
        x, aux = self._forward(
            x, output_logits, remove_logits_bias, output_predictions)
        if self.training and self.aux_logits:
            return x, aux
        else:
            return x


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features, conv_block=None, align_tf=False):
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)
        self.pool_include_padding = not align_tf

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=self.pool_include_padding)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionB, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, stride=2)

    def _forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7, conv_block=None, align_tf=False):
        super(InceptionC, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = conv_block(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = conv_block(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)
        self.pool_include_padding = not align_tf

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=self.pool_include_padding)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionD, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2)

    def _forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels, conv_block=None, align_tf=False, use_max_pool=False):
        super(InceptionE, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)
        self.pool_include_padding = not align_tf
        self.use_max_pool = use_max_pool

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        if self.use_max_pool:
            branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        else:
            branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                       count_include_pad=self.pool_include_padding)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes, conv_block=None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv0 = conv_block(in_channels, 128, kernel_size=1)
        self.conv1 = conv_block(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # N x 768 x 17 x 17
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 768 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 768
        x = self.fc(x)
        # N x 1000
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

# pylint: enable=line-too-long
# pylint: enable=missing-function-docstring
# pylint: enable=missing-class-docstring
# pylint: enable=super-with-arguments
# pylint: enable=consider-merging-isinstance
# pylint: enable=import-outside-toplevel
# pylint: enable=no-else-return
