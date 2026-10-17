#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from abc import ABC, abstractmethod
from typing import Any

import torch


class AbsTokenizer(torch.nn.Module, ABC):
    """The abstract tokenizer class for SpeechLM.

    The main objective of this module is to transform the LM-generated tokens
    into the corresponding targets. E.g.,
    Speech Codec codes -> waveform
    BPE tokens -> text
    ...
    """

    @abstractmethod
    @torch.no_grad()
    def forward(self, tokens: torch.Tensor) -> Any:
        raise NotImplementedError
