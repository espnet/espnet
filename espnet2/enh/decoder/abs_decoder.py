from abc import ABC, abstractmethod
from typing import Tuple

import torch


class AbsDecoder(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @property
    def frame_size(self) -> int:
        raise NotImplementedError

    @property
    def hop_size(self) -> int:
        raise NotImplementedError
    
    def forward_streaming(self, input_frame: torch.Tensor):
        raise NotImplementedError
