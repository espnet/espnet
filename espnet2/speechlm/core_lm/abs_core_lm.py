#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from abc import ABC, abstractmethod
from typing import Tuple, Dict

import torch


class AbsCoreLM(torch.nn.Module, ABC):
    """
    The abstract CoreLM class for SpeechLM.
    It contains:
    (1) Positional Embedding
    (2) Stack Transformer layers
    Nore embedding table is not included in this module.

    Override this module to import LLMs, likc llama2
    """

    @abstractmethod
    def forward(
        self,
        decoder_input: torch.Tensor,
        decoder_input_lengths: torch.Tensor = None,
        encoder_input: torch.Tensor = None,
        encoder_input_lengths: torch.Tensor = None,
        cache: Dict = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        raise NotImplementedError

    @abstractmethod
    def model_dim(self) -> int:
        raise NotImplementedError
