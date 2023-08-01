# python3.7
"""Contains the base class to implement loss."""

import torch

__all__ = ['BaseLoss']


class BaseLoss(object):
    """Base loss class.

    The derived class can easily serialize its members to a `dict`, and to load
    from such `dict` to resume the saved loss.

    NOTE: By default, the derived class will save ALL members. To ensure members
    are saved and loaded as expectation, you may need to override the
    `state_dict()` or `load_state_dict()` method.
    """

    @property
    def name(self):
        """Returns the class name of the loss."""
        return self.__class__.__name__

    def state_dict(self):
        """Returns a serialized `dict` that records all members.

        The returned `dict` maps attribute names to their values.

        NOTE: Override this method if such default behavior is unexpected.
        """
        return vars(self)

    def load_state_dict(self, state_dict):
        """Loads parameters from the `state_dict`.

        By default, this method directly sets all attributes from the given
        `state_dict`.

        NOTE: Override this method if such default behavior is unexpected.
        """
        for key, val in state_dict.items():
            if not hasattr(self, key):  # current loss does not init with `key`
                continue
            origin_attr = getattr(self, key)
            if isinstance(origin_attr, torch.nn.Module):
                origin_attr.load_state_dict(val.state_dict())
                continue
            if isinstance(origin_attr, torch.Tensor):
                val = val.to(device=origin_attr.device)
            setattr(self, key, val)
