#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from typing import Tuple, Dict, List

import torch

from espnet2.speechlm.predictor.abs_predictor import AbsPredictor


class ParallelPredictor(AbsPredictor):
    """parallel pattern in https://arxiv.org/pdf/2306.05284.pdf"""

    def __init__(
        self,
        vocab_size: List,
        input_dim: int,
        nq: int,
    ):
        super(ParallelPredictor, self).__init__()

        self.linear = torch.nn.Linear(input_dim, vocab_size * nq, bias=False)
        self.nq = nq

    def forward(
        self,
        input: torch.Tensor,
        input_lengths: torch.Tensor = None,
        target: torch.Tensor = None,
        target_lengths: torch.Tensor = None,
        cache: dict = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:

        output = self.linear(input)
        B, T, Dnq = output.size()
        output = output.view(B, T, self.nq, Dnq // self.nq)

        return output, input_lengths

    def get_lookup_table(self):
        raise ValueError("Cannot share the lookup table as there are multiple")


class DelayPredictor(ParallelPredictor):
    """delay pattern in https://arxiv.org/pdf/2306.05284.pdf"""

    def organize_target(
        self,
        target: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        raise NotImplementedError
