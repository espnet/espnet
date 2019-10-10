"""PyTorch LR schdulers."""

from typing import Dict

from torch.optim import Optimizer

from espnet.scheduler.scaler import ScalerInterface


class PyTorchScheduler:
    """PyTorch lr scheduler."""

    def __init__(self, scaler_dict: Dict[str, ScalerInterface], optimizer: Optimizer):
        """Initialize class."""
        self.scaler_dict = scaler_dict
        self.optimizer = optimizer
        for k in self.scaler_dict.keys():
            for group in optimizer.param_groups:
                group.setdefault("initial_" + k, group[k])

    def step(self, n_iter: int):
        """Update optimizer by scheduling."""
        for k, scaler in self.scaler_dict.items():
            for group in self.optimizer.param_groups:
                group[k] = group["initial_" + k] * scaler.scale(n_iter)
