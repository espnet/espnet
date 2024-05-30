"""Step (with Warm up) learning rate scheduler module."""

from typing import Union

import torch
from torch.optim.lr_scheduler import _LRScheduler
from typeguard import typechecked

from espnet2.schedulers.abs_scheduler import AbsBatchStepScheduler


class WarmupStepLR(_LRScheduler, AbsBatchStepScheduler):
    """The WarmupStepLR scheduler.

    This scheduler is the combination of WarmupLR and StepLR:

    WarmupLR:
        lr = optimizer.lr * warmup_step ** 0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    WarmupStepLR:
        if step <= warmup_step:
            lr = optimizer.lr * warmup_step ** 0.5
                 * min(step ** -0.5, step * warmup_step ** -1.5)
        else:
            lr = optimizer.lr * (gamma ** (epoch//step_size))

    Note that the maximum lr equals to optimizer.lr in this scheduler.

    """

    @typechecked
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        # for WarmupLR
        warmup_steps: Union[int, float] = 25000,
        # for StepLR
        steps_per_epoch: int = 10000,
        step_size: int = 1,
        gamma: float = 0.1,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps

        self.step_num = 0
        self.epoch_num = 0
        # NOTE: This number should be adjusted accordingly
        #       once batch_size/ngpu/num_nodes is changed.
        # To get the exact number of iterations per epoch, refer to
        # https://github.com/espnet/espnet/discussions/4404
        self.steps_per_epoch = steps_per_epoch
        self.warmup_epoch = warmup_steps // steps_per_epoch

        self.lr_scale = warmup_steps**-1

        # after warmup_steps, decrease lr by `gamma` every `step_size` epochs
        self.step_size = step_size
        self.gamma = gamma

        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(warmup_steps={self.warmup_steps}, "
            f"steps_per_epoch={self.steps_per_epoch},"
            f" step_size={self.step_size}, gamma={self.gamma})"
        )

    def get_lr(self):
        self.step_num += 1
        if self.step_num % self.steps_per_epoch == 0:
            self.epoch_num += 1

        if self.step_num <= self.warmup_steps:
            return [lr * self.lr_scale * self.step_num for lr in self.base_lrs]
        else:
            return [
                lr
                * self.gamma ** ((self.epoch_num - self.warmup_epoch) // self.step_size)
                for lr in self.base_lrs
            ]
