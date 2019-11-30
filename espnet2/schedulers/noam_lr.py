from typing import Union

import torch
from torch.optim.lr_scheduler import LambdaLR
from typeguard import check_argument_types


class NoamLR(LambdaLR):
    def __init__(self, optimizer: torch.optim.Optimizer,
                 model_size: Union[int, float] = 320,
                 warmup_steps: Union[int, float] = 25000,
                 last_epoch: int = -1):
        assert check_argument_types()
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, lr_lambda=self._lr_lambda,
                         last_epoch=last_epoch)

    def _lr_lambda(self, last_epoch):
        last_epoch = max(1, last_epoch)
        scale = self.model_size ** -0.5 * \
            min(last_epoch ** -0.5, last_epoch * self.warmup_steps ** -1.5)
        return scale
