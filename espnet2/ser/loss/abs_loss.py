# code from https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
# Adapted from https://github.com/wujiyang/Face_Pytorch (Apache License)
from abc import abstractmethod

import torch
import torch.nn as nn


class AbsLoss(nn.Module):
    def __init__(self, nout: int, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, label=None) -> torch.Tensor:
        raise NotImplementedError
