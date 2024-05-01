from abc import ABC, abstractmethod
from typing import Tuple, Dict

import torch


class AbsPostProcessor(torch.nn.Module, ABC):
    """
    The abstract Post-Processor class for SpeechLM.
    The main objective of this module is to transform the LM-generated tokens
    into the corresponding targets. E.g.,
    Speech Codec codes -> waveform
    BPE tokens -> text
    ...
    """

    @abstractmethod
    @torch.no_grad()
    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        raise NotImplementedError
