from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch


class AbsSeparator(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[Tuple[torch.Tensor], torch.Tensor, OrderedDict]:
        raise NotImplementedError
    
    @abstractmethod
    def forward_streaming(
        self,
        input_frame: torch.Tensor,
        buffer = None,
    ):
        raise NotImplementedError
        

    @property
    @abstractmethod
    def num_spk(self):
        raise NotImplementedError
