# This script is copied from
# https://github.com/Lightning-AI/pytorch-lightning/issues/3346#issuecomment-1478556073

from typing import Dict, Iterable, List, Union, Any
import torch
from typeguard import typechecked

from espnetez.trainer.hybrid_optim import HybridOptim


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
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        optimizer_idx: int,
    ) -> None:
        self.optimizer = hybrid_optimizer
        self.lr_scheduler = lr_scheduler
        self.idx = optimizer_idx

    def __getattribute__(self, __name: str) -> Any:
        """
        If the attribute name is one of ``optimizer``, ``idx``, or ``lr_scheduler``, return this
        class's attribute with the same name, else return the ``lr_scheduler``'s attribute with
        that name.

        """
        if __name in {"optimizer", "lr_scheduler", "idx"}:
            return super().__getattribute__(__name)
        else:
            return self.lr_scheduler.__getattribute__(__name)
