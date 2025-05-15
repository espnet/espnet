# This script is copied from
# https://github.com/Lightning-AI/pytorch-lightning/issues/3346#issuecomment-1478556073

from typing import Any, Dict, Iterable, List, Union

import torch
from typeguard import typechecked

from espnet3.trainer.hybrid_optim import HybridOptim


class HybridLRS(torch.optim.lr_scheduler._LRScheduler):
    """
    Wrapper class around ``lr_scheduler``s to return a dummy optimizer to pass PyTorch Lightning
    checks.

    Modified from the reply in a GitHub Issue thread here:
    https://github.com/Lightning-AI/lightning/issues/3346#issuecomment-1036063687

    Parameters
    ----------
    hybrid_optimizer: HybridOptim
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler
    idx: int
        Index of the optimizer in ``hybrid_optimizer`` the learning rate scheduler ``lr_scheduler``
        is assigned to

    """

    @typechecked
    def __init__(
        self,
        hybrid_optimizer: HybridOptim,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        optimizer_idx: int,
    ) -> None:
        self.optimizer = hybrid_optimizer
        self.lr_scheduler = lr_scheduler
        self.idx = optimizer_idx

    def __getattr__(self, name: str) -> Any:
        if name in {"optimizer", "lr_scheduler", "idx"}:
            return getattr(self, name)
        else:
            return self.lr_scheduler.__getattribute__(name)
