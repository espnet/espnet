"""Linear warmup then linear decay learning rate scheduler module."""

from typing import Union

import torch
from torch.optim.lr_scheduler import _LRScheduler
from typeguard import typechecked

from espnet2.schedulers.abs_scheduler import AbsBatchStepScheduler


class LinearWarmupDecayLR(_LRScheduler, AbsBatchStepScheduler):
    """The LinearWarmupDecayLR scheduler

    This reproduces F5-TTS's training schedule
    (https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/model/trainer.py):
    a linear warmup ``start_factor -> 1.0`` over ``warmup_steps`` followed by a
    linear decay ``1.0 -> end_factor`` over the remaining ``total_steps -
    warmup_steps`` updates. Stepped per optimizer update
    (``scheduler_interval: step``).

    Unlike ESPnet's ``WarmupLR`` (inverse-sqrt, non-terminating), the linear
    decay needs to know the total number of updates, so ``total_steps`` must be
    set to the planned training length (``epochs * updates_per_epoch``),
    matching how upstream computes ``decay_updates = total_updates -
    warmup_updates``. Once ``total_steps`` is passed the learning rate is
    clamped at ``end_factor * base_lr``; it never rises again.
    """

    @typechecked
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: Union[int, float] = 20000,
        total_steps: Union[int, float] = 600000,
        start_factor: float = 1e-8,
        end_factor: float = 1e-8,
        last_epoch: int = -1,
    ):
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.decay_steps = max(self.total_steps - self.warmup_steps, 1)

        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(warmup_steps={self.warmup_steps}, "
            f"total_steps={self.total_steps}, "
            f"start_factor={self.start_factor}, "
            f"end_factor={self.end_factor})"
        )

    def get_lr(self):
        step_num = self.last_epoch
        param_groups = self.optimizer.param_groups

        # Initial step: scale the base lr down to the warmup floor.
        if step_num == 0:
            return [group["lr"] * self.start_factor for group in param_groups]

        # Warmup phase: start_factor -> 1.0 over warmup_steps updates.
        if step_num < self.warmup_steps:
            delta = 1.0 - self.start_factor
            factor = 1.0 + delta / (
                self.warmup_steps * self.start_factor + (step_num - 1) * delta
            )
            return [group["lr"] * factor for group in param_groups]

        # Handover: the peak is exactly the base lr, with no accumulated drift.
        if step_num == self.warmup_steps:
            return list(self.base_lrs)

        # Decay phase: 1.0 -> end_factor over decay_steps updates, then clamp.
        decay_num = step_num - self.warmup_steps
        if decay_num > self.decay_steps:
            return [group["lr"] for group in param_groups]
        delta = self.end_factor - 1.0
        factor = 1.0 + delta / (self.decay_steps * 1.0 + (decay_num - 1) * delta)
        return [group["lr"] * factor for group in param_groups]


def linear_warmup_decay(
    optimizer,
    warmup_steps: int,
    total_steps: int,
    start_factor: float = 1e-8,
    end_factor: float = 1e-8,
):
    """Build a LinearWarmupDecayLR.

    Retained for compatibility with configs that target this factory by its
    dotted path. New configs should target ``LinearWarmupDecayLR`` directly.
    """
    return LinearWarmupDecayLR(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        start_factor=start_factor,
        end_factor=end_factor,
    )
