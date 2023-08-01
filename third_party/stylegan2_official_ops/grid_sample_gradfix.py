# python3.7

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Custom replacement for `torch.nn.functional.grid_sample`.

This is useful for differentiable augmentation. This customized operator
supports arbitrarily high order gradients between the input and output. Only
works on 2D images and assumes `mode=bilinear`, `padding_mode=zeros`, and
`align_corners=False`.

Please refer to https://github.com/NVlabs/stylegan2-ada-pytorch
"""

# pylint: disable=redefined-builtin
# pylint: disable=arguments-differ
# pylint: disable=protected-access
# pylint: disable=line-too-long
# pylint: disable=missing-function-docstring

import warnings
import torch
from distutils.version import LooseVersion

_use_pytorch_1_11_api = LooseVersion(torch.__version__) >= LooseVersion('1.11.0')
#----------------------------------------------------------------------------

enabled = True  # Enable the custom op by setting this to true.

#----------------------------------------------------------------------------

def grid_sample(input, grid, impl='cuda'):
    if impl == 'cuda' and _should_use_custom_op():
        return _GridSample2dForward.apply(input, grid)
    return torch.nn.functional.grid_sample(input=input, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=False)

#----------------------------------------------------------------------------

def _should_use_custom_op():
    if not enabled:
        return False
    if LooseVersion(torch.__version__) >= LooseVersion('1.7.0'):
        return True
    warnings.warn(f'grid_sample_gradfix not supported on PyTorch {torch.__version__}. Falling back to torch.nn.functional.grid_sample().')
    return False

#----------------------------------------------------------------------------

class _GridSample2dForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid):
        assert input.ndim == 4
        assert grid.ndim == 4
        output = torch.nn.functional.grid_sample(input=input, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        ctx.save_for_backward(input, grid)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        grad_input, grad_grid = _GridSample2dBackward.apply(grad_output, input, grid)
        return grad_input, grad_grid

#----------------------------------------------------------------------------

class _GridSample2dBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, input, grid):
        op = torch._C._jit_get_operation('aten::grid_sampler_2d_backward')
        if _use_pytorch_1_11_api:
            output_mask = (ctx.needs_input_grad[1], ctx.needs_input_grad[2])
            grad_input, grad_grid = op(grad_output, input, grid, 0, 0, False, output_mask)
        else:
            grad_input, grad_grid = op(grad_output, input, grid, 0, 0, False)
        ctx.save_for_backward(grid)
        return grad_input, grad_grid

    @staticmethod
    def backward(ctx, grad2_grad_input, grad2_grad_grid):
        _ = grad2_grad_grid # unused
        grid, = ctx.saved_tensors
        grad2_grad_output = None
        grad2_input = None
        grad2_grid = None

        if ctx.needs_input_grad[0]:
            grad2_grad_output = _GridSample2dForward.apply(grad2_grad_input, grid)

        assert not ctx.needs_input_grad[2]
        return grad2_grad_output, grad2_input, grad2_grid

#----------------------------------------------------------------------------

# pylint: enable=redefined-builtin
# pylint: enable=arguments-differ
# pylint: enable=protected-access
# pylint: enable=line-too-long
# pylint: enable=missing-function-docstring
