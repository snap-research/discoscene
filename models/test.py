# python3.7
"""Unit test for loading pre-trained models.

Basically, this file tests whether the perceptual model (VGG16) and the
inception model (InceptionV3), which are commonly used for loss computation and
evaluation, have the expected behavior after loading pre-trained weights. In
particular, we compare with the models from repo

https://github.com/NVlabs/stylegan2-ada-pytorch
"""

import torch

from models import build_model
from utils.misc import download_url

__all__ = ['test_model']

_BATCH_SIZE = 4
# pylint: disable=line-too-long
_PERCEPTUAL_URL = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
_INCEPTION_URL = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
# pylint: enable=line-too-long


def test_model():
    """Collects all model tests."""
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print('========== Start Model Test ==========')
    test_perceptual()
    test_inception()
    print('========== Finish Model Test ==========')


def test_perceptual():
    """Test the perceptual model."""
    print('===== Testing Perceptual Model =====')

    print('Build test model.')
    model = build_model('PerceptualModel',
                        use_torchvision=False,
                        no_top=False,
                        enable_lpips=True)

    print('Build reference model.')
    ref_model_path, _, = download_url(_PERCEPTUAL_URL)
    with open(ref_model_path, 'rb') as f:
        ref_model = torch.jit.load(f).eval().cuda()

    print('Test performance: ')
    for size in [224, 128, 256, 512, 1024]:
        raw_img = torch.randint(0, 256, size=(_BATCH_SIZE, 3, size, size))
        raw_img_comp = torch.randint(0, 256, size=(_BATCH_SIZE, 3, size, size))

        # The test model requires input images to have range [-1, 1].
        img = raw_img.to(torch.float32).cuda() / 127.5 - 1
        img_comp = raw_img_comp.to(torch.float32).cuda() / 127.5 - 1
        feat = model(img, resize_input=True, return_tensor='feature')
        pred = model(img, resize_input=True, return_tensor='prediction')
        lpips = model(img, img_comp, resize_input=False, return_tensor='lpips')
        assert feat.shape == (_BATCH_SIZE, 4096)
        assert pred.shape == (_BATCH_SIZE, 1000)
        assert lpips.shape == (_BATCH_SIZE,)

        # The reference model requires input images to have range [0, 255].
        img = raw_img.to(torch.float32).cuda()
        img_comp = raw_img_comp.to(torch.float32).cuda()
        ref_feat = ref_model(img, resize_images=True, return_features=True)
        ref_pred = ref_model(img, resize_images=True, return_features=False)
        temp = ref_model(torch.cat([img, img_comp], dim=0),
                         resize_images=False, return_lpips=True).chunk(2)
        ref_lpips = (temp[0] - temp[1]).square().sum(dim=1, keepdim=False)
        assert ref_feat.shape == (_BATCH_SIZE, 4096)
        assert ref_pred.shape == (_BATCH_SIZE, 1000)
        assert ref_lpips.shape == (_BATCH_SIZE,)

        print(f'    Size {size}x{size}, feature (with resize):\n        '
              f'mean: {(feat - ref_feat).abs().mean().item():.3e}, '
              f'max: {(feat - ref_feat).abs().max().item():.3e}, '
              f'ref_mean: {ref_feat.abs().mean().item():.3e}, '
              f'ref_max: {ref_feat.abs().max().item():.3e}.')
        print(f'    Size {size}x{size}, prediction (with resize):\n        '
              f'mean: {(pred - ref_pred).abs().mean().item():.3e}, '
              f'max: {(pred - ref_pred).abs().max().item():.3e}, '
              f'ref_mean: {ref_pred.abs().mean().item():.3e}, '
              f'ref_max: {ref_pred.abs().max().item():.3e}.')
        print(f'    Size {size}x{size}, LPIPS (without resize):\n        '
              f'mean: {(lpips - ref_lpips).abs().mean().item():.3e}, '
              f'max: {(lpips - ref_lpips).abs().max().item():.3e}, '
              f'ref_mean: {ref_lpips.abs().mean().item():.3e}, '
              f'ref_max: {ref_lpips.abs().max().item():.3e}.')


def test_inception():
    """Test the inception model."""
    print('===== Testing Inception Model =====')

    print('Build test model.')
    model = build_model('InceptionModel', align_tf=True)

    print('Build reference model.')
    ref_model_path, _, = download_url(_INCEPTION_URL)
    with open(ref_model_path, 'rb') as f:
        ref_model = torch.jit.load(f).eval().cuda()

    print('Test performance: ')
    for size in [299, 128, 256, 512, 1024]:
        raw_img = torch.randint(0, 256, size=(_BATCH_SIZE, 3, size, size))

        # The test model requires input images to have range [-1, 1].
        img = raw_img.to(torch.float32).cuda() / 127.5 - 1
        feat = model(img)
        pred = model(img, output_predictions=True)
        pred_nb = model(img, output_predictions=True, remove_logits_bias=True)
        assert feat.shape == (_BATCH_SIZE, 2048)
        assert pred.shape == (_BATCH_SIZE, 1008)
        assert pred_nb.shape == (_BATCH_SIZE, 1008)

        # The reference model requires input images to have range [0, 255].
        img = raw_img.to(torch.float32).cuda()
        ref_feat = ref_model(img, return_features=True)
        ref_pred = ref_model(img)
        ref_pred_nb = ref_model(img, no_output_bias=True)
        assert ref_feat.shape == (_BATCH_SIZE, 2048)
        assert ref_pred.shape == (_BATCH_SIZE, 1008)
        assert ref_pred_nb.shape == (_BATCH_SIZE, 1008)

        print(f'    Size {size}x{size}, feature:\n        '
              f'mean: {(feat - ref_feat).abs().mean().item():.3e}, '
              f'max: {(feat - ref_feat).abs().max().item():.3e}, '
              f'ref_mean: {ref_feat.abs().mean().item():.3e}, '
              f'ref_max: {ref_feat.abs().max().item():.3e}.')
        print(f'    Size {size}x{size}, prediction:\n        '
              f'mean: {(pred - ref_pred).abs().mean().item():.3e}, '
              f'max: {(pred - ref_pred).abs().max().item():.3e}, '
              f'ref_mean: {ref_pred.abs().mean().item():.3e}, '
              f'ref_max: {ref_pred.abs().max().item():.3e}.')
        print(f'    Size {size}x{size}, prediction (without bias):\n        '
            f'mean: {(pred_nb - ref_pred_nb).abs().mean().item():.3e}, '
            f'max: {(pred_nb - ref_pred_nb).abs().max().item():.3e}, '
              f'ref_mean: {ref_pred_nb.abs().mean().item():.3e}, '
              f'ref_max: {ref_pred_nb.abs().max().item():.3e}.')
