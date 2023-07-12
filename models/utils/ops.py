# python3.7
"""Contains operators for neural networks."""

import torch
import torch.distributed as dist

__all__ = ['all_gather']


def all_gather(tensor):
    """Gathers tensor from all devices and executes averaging."""
    if not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()
    tensor_list = [torch.ones_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor, async_op=False)
    return torch.stack(tensor_list, dim=0).mean(dim=0)
