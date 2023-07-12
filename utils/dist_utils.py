# python3.7
"""Contains utility functions used for distribution."""

import contextlib
import os
import subprocess

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

__all__ = ['init_dist', 'exit_dist', 'ddp_sync', 'get_ddp_module']


def init_dist(launcher, backend='nccl', **kwargs):
    """Initializes distributed environment."""
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)
        dist.init_process_group(backend=backend, **kwargs)
    elif launcher == 'slurm':
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(proc_id % num_gpus)
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        port = os.environ.get('PORT', 29500)
        os.environ['MASTER_PORT'] = str(port)
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        dist.init_process_group(backend=backend)
    else:
        raise NotImplementedError(f'Not implemented launcher type: '
                                  f'`{launcher}`!')


def exit_dist():
    """Exits the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


@contextlib.contextmanager
def ddp_sync(model, sync):
    """Controls whether the `DistributedDataParallel` model should be synced."""
    assert isinstance(model, torch.nn.Module)
    is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
    if sync or not is_ddp:
        yield
    else:
        with model.no_sync():
            yield


def get_ddp_module(model):
    """Gets the module from `DistributedDataParallel`."""
    assert isinstance(model, torch.nn.Module)
    is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
    if is_ddp:
        return model.module
    return model
