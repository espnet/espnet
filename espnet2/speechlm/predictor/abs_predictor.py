from abc import ABC, abstractmethod
from typing import Tuple, Dict, List

import torch


class AbsPredictor(torch.nn.Module, ABC):
    """ The abstract Speech LM class """

    @abstractmethod
    def forward(
        self, 
        input: torch.Tensor, 
        input_lengths: torch.Tensor = None,
        cache: dict = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        raise NotImplementedError

    @abstractmethod
    def organize_target(
        self, 
        target: torch.Tensor, 
        target_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
    
    @abstractmethod
    def get_lookup_table(self):
        raise NotImplementedError