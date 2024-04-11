from abc import ABC, abstractmethod
from typing import Tuple, Dict

import torch


class AbsCoreLM(torch.nn.Module, ABC):
    """
    The abstract CoreLM class for SpeechLM.
    The coreLM are generally the stacked Transformer layers from many sources.
    It can specifically support many pre-trained models such as LLaMA 2, Gemma etc.
    Note this module doesn't include embedding / lm_head layers
    """

    @abstractmethod
    def forward(
        self, input: torch.Tensor, input_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        raise NotImplementedError

    @abstractmethod
    def inference(
        self, prefix: torch.Tensor, input_mask: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        raise NotImplementedError
