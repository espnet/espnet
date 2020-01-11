from distutils.version import LooseVersion
from typing import Union

import torch
from torch.optim.lr_scheduler import _LRScheduler
from typeguard import check_argument_types

from espnet2.schedulers.abs_scheduler import AbsBatchStepScheduler


class NoamLR(_LRScheduler, AbsBatchStepScheduler):
    """The LR scheduler proposed by Noam

    FIXME(kamo): PyTorch doesn't provide _LRScheduler as public class,
     thus the behaviour isn't guaranteed at forward PyTorch version.

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: Union[int, float] = 25000,
        last_epoch: int = -1,
    ):
        if LooseVersion(torch.__version__) < LooseVersion("1.1.0"):
            raise NotImplementedError(f"Require PyTorch>=1.1.0: {torch.__version__}")

        assert check_argument_types()
        self.warmup_steps = warmup_steps

        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = self.last_epoch + 1
        return [
            lr * min(step_num ** -0.5, step_num * self.warmup_steps ** -1.5)
            for lr in self.base_lrs
        ]
