"""Chainer optimizer schdulers."""

import logging
from typing import List

from espnet.scheduler.scheduler import SchedulerInterface

try:
    from chainer.optimizer import Optimizer
except ImportError:
    logging.warning("Chainer is not Installed. Run `make chainer.done` at tools dir.")


class ChainerScheduler:
    """Chainer optimizer scheduler."""

    def __init__(self, schedulers: List[SchedulerInterface], optimizer: "Optimizer"):
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
