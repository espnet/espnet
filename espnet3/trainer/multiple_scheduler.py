"""Multiple scheduler for multiple optimizers in Lightning."""

# This script is copied from
# https://github.com/Lightning-AI/pytorch-lightning/issues/3346#issuecomment-1478556073

from typing import Any

import torch
from typeguard import typechecked

from espnet3.trainer.multiple_optim import MultipleOptim


class MultipleScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Wrapper class around ``lr_scheduler``s to return a dummy optimizer.

    Modified from the reply in a GitHub Issue thread here:
    https://github.com/Lightning-AI/lightning/issues/3346#issuecomment-1036063687

    Args:
        multiple_optim: MultipleOptim
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler
        optimizer_idx: int
            Index of the optimizer in ``multiple_optimizer`` the learning rate scheduler
            ``lr_scheduler`` is assigned to

    """

    @typechecked
    def __init__(
        self,
        multiple_optimizer: MultipleOptim,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        optimizer_idx: int,
    ) -> None:
        """Initialize MultipleScheduler object."""
        assert 0 <= optimizer_idx < len(multiple_optimizer.optimizers), (
            f"optimizer_idx {optimizer_idx} is out of range for "
            f"multiple_optimizer with {len(multiple_optimizer.optimizers)} optimizers."
        )
        self.optimizer = multiple_optimizer
        self.lr_scheduler = lr_scheduler
        self.idx = optimizer_idx

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to either self or self.lr_scheduler."""
        if name in {"optimizer", "lr_scheduler", "idx"}:
            return getattr(self, name)
        else:
            return self.lr_scheduler.__getattribute__(name)
