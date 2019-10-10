"""PyTorch LR schdulers."""

from typing import List

from torch.optim import Optimizer

from espnet.scheduler.scaler import ScalerInterface


class PyTorchScheduler:
    """PyTorch lr scheduler."""

    def __init__(self, scalers: List[ScalerInterface], optimizer: Optimizer):
        """Initialize class."""
        self.scalers = scalers
        self.optimizer = optimizer
        for s in self.scalers:
            for group in optimizer.param_groups:
                group.setdefault("initial_" + s.key, group[s.key])

    def step(self, n_iter: int):
        """Update optimizer by scheduling."""
        for s in self.scalers:
            for group in self.optimizer.param_groups:
                group[s.key] = group["initial_" + s.key] * s.scale(n_iter)
