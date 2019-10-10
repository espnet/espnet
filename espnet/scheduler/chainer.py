"""Chainer optimizer schdulers."""

from typing import List

from chainer.optimizer import Optimizer

from espnet.scheduler.scaler import ScalerInterface


class ChainerScheduler:
    """Chainer optimizer scheduler."""

    def __init__(self, scalers: List[ScalerInterface], optimizer: Optimizer):
        """Initialize class."""
        self.scalers = scalers
        self.optimizer = optimizer
        self.init_values = dict()
        for s in self.scalers:
            self.init_values[s.key] = getattr(self.optimizer, s.key)

    def step(self, n_iter: int):
        """Update optimizer by scheduling."""
        for s in self.scalers:
            new_val = self.init_values[s.key] * s.scale(n_iter)
            setattr(self.optimizer, s.key, new_val)
