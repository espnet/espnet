from abc import ABC
from abc import abstractmethod
from typing import List, Tuple, Dict

import torch

from espnet2.enh.loss.criterions.abs_loss import AbsEnhLoss


class AbsLossWrapper(torch.nn.Module,ABC):
    
    criterion: AbsEnhLoss = None

    @abstractmethod
    def forward(
        self,
        ref: List,
        inf: List,
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        raise NotImplementedError