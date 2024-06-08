#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from abc import ABC, abstractmethod
from typing import Any

import torch


class AbsTokenizer(torch.nn.Module, ABC):
    """Abstract SpeechLM tokenizer class"""

    @abstractmethod
    @torch.no_grad()
    def forward(self, inp: Any) -> torch.Tensor:
        """Tokenization function"""
        raise NotImplementedError

    @torch.no_grad()
    def detokenize(self, tokens: torch.Tensor) -> Any:
        """Detokenization function"""
        raise NotImplementedError
