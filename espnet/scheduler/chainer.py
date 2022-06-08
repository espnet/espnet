"""Chainer optimizer schdulers."""

from typing import List

from chainer.optimizer import Optimizer

from espnet.scheduler.scheduler import SchedulerInterface


class ChainerScheduler:
    """Chainer optimizer scheduler."""

    def __init__(self, schedulers: List[SchedulerInterface], optimizer: Optimizer):
        """Initialize class."""
        self.schedulers = schedulers
        self.optimizer = optimizer
        self.init_values = dict()
        for s in self.schedulers:
            self.init_values[s.key] = getattr(self.optimizer, s.key)

    def step(self, n_iter: int):
        """Update optimizer by scheduling."""
        for s in self.schedulers:
            new_val = self.init_values[s.key] * s.scale(n_iter)
            setattr(self.optimizer, s.key, new_val)
