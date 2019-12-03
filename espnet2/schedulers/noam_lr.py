from typing import Union

import torch
from torch.optim.lr_scheduler import _LRScheduler
from typeguard import check_argument_types

from espnet2.schedulers.abs_scheduler import AbsBatchScheduler


class NoamLR(_LRScheduler, AbsBatchScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer,
                 model_size: Union[int, float] = 320,
                 warmup_steps: Union[int, float] = 25000,
                 last_epoch: int = -1):
        assert check_argument_types()
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [lr * self.model_size ** -0.5 *
                min(self._step_count ** -0.5,
                    self._step_count * self.warmup_steps ** -1.5)
                for lr in self.base_lrs]
