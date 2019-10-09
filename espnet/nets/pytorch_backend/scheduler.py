"""PyTorch schdulers.

TODO(karita): implement new own LambdaLR
"""

from espnet.nets.scheduler_interface import SchedulerInterface

import torch


class PyTorchScheduler:
    """PyTorch scheduler."""

    def __init__(self, scheduler: SchedulerInterface, optimizer: torch.optim.Optimizer):
        """Initialize class."""
        self.scheduler = scheduler
        self.optimizer = optimizer
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])

    def step(self, n_iter: int):
        """Update optimizer by scheduling."""
        for group in self.optimizer.param_groups:
            group["lr"] = group["initial_lr"] * self.scheduler.scale(n_iter)
