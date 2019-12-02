from abc import ABC, abstractmethod

import torch
from typing import Tuple


class InversibleInterface(ABC):
    @abstractmethod
    def inverse(self, input: torch.Tensor, input_lengths: torch.Tensor = None)\
            -> Tuple[torch.Tensor, torch.Tensor]:
        # return output, output_lengths
        raise NotImplementedError
