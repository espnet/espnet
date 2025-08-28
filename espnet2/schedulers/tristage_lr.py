import math
from typing import Union

import torch
from torch.optim.lr_scheduler import _LRScheduler
from typeguard import typechecked

from espnet2.schedulers.abs_scheduler import AbsBatchStepScheduler


class TristageLR(_LRScheduler, AbsBatchStepScheduler):
    """Tri-stage learning rate scheduler with warmup, hold, and exponential decay.

    This scheduler adjusts the learning rate in three phases:
        1. Warmup:
           The learning rate increases linearly from
           `init_lr_scale * base_lr` to `base_lr` over the first
           `warmup_ratio * max_steps` steps.

        2. Hold:
           The learning rate is held constant at `base_lr` for
           `hold_ratio * max_steps` steps.

        3. Decay:
           The learning rate decays exponentially from `base_lr`
           to `final_lr_scale * base_lr` over `decay_ratio * max_steps`
           steps.

    Reference:
        Adapted from the tri-stage LR scheduler in fairseq:
        https://github.com/facebookresearch/fairseq/blob/main/fairseq/
        optim/lr_scheduler/tri_stage_lr_scheduler.py

    Args:
        optimizer: Wrapped optimizer.
        max_steps: Total number of steps.
        warmup_ratio: Fraction of steps for linear warmup.
        hold_ratio: Fraction of steps to hold constant.
        decay_ratio: Fraction of steps for exponential decay.
        init_lr_scale: Initial learning rate is `init_lr_scale * base_lr`.
        final_lr_scale: Final learning rate is `final_lr_scale * base_lr`.
        last_epoch: The index of the last step. Default is -1 (fresh start).
    """

    @typechecked
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_steps: Union[int, float] = 25000,
        warmup_ratio: float = 0.1,
        hold_ratio: float = 0.4,
        decay_ratio: float = 0.5,
        init_lr_scale: float = 0.01,
        final_lr_scale: float = 0.01,
        last_epoch: int = -1,
    ):
        assert math.isclose(
            warmup_ratio + hold_ratio + decay_ratio, 1.0
        ), "The sum of warmup_ratio, hold_ratio, and decay_ratio must be 1.0."
        assert 0.0 < init_lr_scale <= 1.0, "init_lr_scale must be in (0, 1]."
        assert 0.0 < final_lr_scale <= 1.0, "final_lr_scale must be in (0, 1]."
        self.max_steps = int(max_steps)
        self.warmup_steps = int(self.max_steps * warmup_ratio)
        self.hold_steps = int(self.max_steps * hold_ratio)
        self.decay_steps = self.max_steps - self.warmup_steps - self.hold_steps
        self.init_lr_scale = init_lr_scale
        self.final_lr_scale = final_lr_scale
        if self.decay_steps != 0:
            self.decay_factor = -math.log(final_lr_scale) / self.decay_steps
        else:
            self.decay_factor = 0.0

        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        express = f"{self.__class__.__name__}(warmup_steps={self.warmup_steps})"
        express += f"(hold_steps={self.hold_steps})"
        express += f"(decay_steps={self.decay_steps})"
        express += f"(init_lr_scale={self.init_lr_scale})"
        express += f"(final_lr_scale={self.final_lr_scale})"
        express += f"(decay_factor={self.decay_factor})"
        return express

    def get_lr(self):

        step_num = self.last_epoch + 1
        step_num = min(step_num, self.max_steps)
        if step_num < self.warmup_steps:
            return [
                self.init_lr_scale * base_lr
                + (base_lr - self.init_lr_scale * base_lr)
                / self.warmup_steps
                * step_num
                for base_lr in self.base_lrs
            ]
        elif step_num < self.warmup_steps + self.hold_steps:
            return [base_lr for base_lr in self.base_lrs]
        else:
            return [
                base_lr
                * math.exp(
                    -self.decay_factor
                    * (step_num - self.warmup_steps - self.hold_steps)
                )
                for base_lr in self.base_lrs
            ]
