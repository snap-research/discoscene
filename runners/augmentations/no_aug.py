# python3.7
"""Defines the dummy augmentation pipeline that executes no augmentation."""

import torch.nn as nn

__all__ = ['NoAug']


NoAug = nn.Identity
