from abc import ABC, abstractmethod
from typing import Tuple, Dict

import torch

from espnet2.speechlm.postprocessor.abs_postprocessor import AbsPostProcessor


class CodecPostProcessor(AbsPostProcessor):
    """The abstract Post-Processor class for SpeechLM"""

    def forward(
        self, input: torch.Tensor, input_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        raise NotImplementedError
