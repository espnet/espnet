"""PyTorch optimizer schdulers."""

from typing import List

from torch.optim import Optimizer

from espnet.scheduler.scheduler import SchedulerInterface


class PyTorchScheduler:
    """PyTorch optimizer scheduler."""

    def __init__(self, schedulers: List[SchedulerInterface], optimizer: Optimizer):
        """Initialize class."""
        self.schedulers = schedulers
        self.optimizer = optimizer
        for s in self.schedulers:
            for group in optimizer.param_groups:
                group.setdefault("initial_" + s.key, group[s.key])

    def step(self, n_iter: int):
        """Update optimizer by scheduling."""
        for s in self.schedulers:
            for group in self.optimizer.param_groups:
                group[s.key] = group["initial_" + s.key] * s.scale(n_iter)
