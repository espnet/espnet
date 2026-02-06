"""Exponential decrease learning rate scheduler module."""

import math

import torch
from torch.optim.lr_scheduler import _LRScheduler

from espnet2.schedulers.abs_scheduler import AbsBatchStepScheduler


class ExponentialDecayWarmup(_LRScheduler, AbsBatchStepScheduler):
    """Exponential Decay with Warmup.

    if step < warmup_steps:
        if warm_from_zero:
            lr = initial_lr * (step / warmup_steps)
        else:
            lr = initial_lr
        else:
            decay_factor = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = initial_lr * exp(decay_factor * log(final_lr / initial_lr))

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_lr (float): Initial learning rate (before decay).
        min_lr (float): Final learning rate (after decay).
        total_steps (int): Total number of steps (epochs * iters per epoch).
        warmup_steps (int): Number of warmup steps. Default: 0.
        warm_from_zero (bool): If True, warmup starts from 0 to initial_lr.
        last_epoch (int): The index of last step. Default: -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        min_lr: float,
        total_steps: int,
        warmup_steps: int = 0,
        warm_from_zero: bool = False,
        last_epoch: int = -1,
    ):
        assert warmup_steps < total_steps

        self.total_steps = total_steps
        self.initial_lr = max_lr
        self.final_lr = min_lr
        self.warmup_steps = warmup_steps
        self.warm_from_zero = warm_from_zero

        # Precompute log decay factor
        self.log_decay = math.log(min_lr / max_lr)

        self.base_lrs = [max_lr for _ in optimizer.param_groups]

        super(ExponentialDecayWarmup, self).__init__(optimizer, last_epoch)

        self.init_lr()

    def init_lr(self):
        for param_group in self.optimizer.param_groups:
            if self.warm_from_zero:
                param_group["lr"] = 0.0
            else:
                param_group["lr"] = self.initial_lr

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warmup phase
            if self.warm_from_zero:
                warmup_factor = (self.last_epoch + 1) / self.warmup_steps
                return [self.initial_lr * warmup_factor for _ in self.base_lrs]
            else:
                return self.base_lrs
        else:
            # Exponential decay phase
            decay_step = self.last_epoch - self.warmup_steps
            decay_total = self.total_steps - self.warmup_steps
            decay_factor = math.exp(self.log_decay * decay_step / decay_total)
            return [self.initial_lr * decay_factor for _ in self.base_lrs]

    def step(self, epoch: int = None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr
