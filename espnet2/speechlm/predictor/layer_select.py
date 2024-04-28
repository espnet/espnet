#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from typing import Tuple, Dict, List

import torch

from espnet2.speechlm.predictor.abs_predictor import AbsPredictor


class LayerSelectPredictor(AbsPredictor):
    """ 
    Predictor for AR-NAR model, 
    specifically Vall-E: https://arxiv.org/abs/2301.02111.
    """

    def __init__(
        self,
        vocab_size: List,
        input_dim: int,
        nq: int,
    ):
        super(LayerSelectPredictor, self).__init__()

        self.lm_head = torch.nn.Linear(input_dim, vocab_size, bias=False)
        self.nq = nq

    def get_lookup_table(self):
        return self.lm_head.weight

    def forward(
        self,
        input: torch.Tensor,
        input_lengths: torch.Tensor,
        target: torch.Tensor,
        target_lengths: torch.Tensor,
        others: dict = None,
        cache: dict = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        assert input.size(0) == target.size(0)
        assert input.size(1) == target.size(1)
        assert input.size(3) == target.size(3)
        assert input.size(2) == 2 # one for AR; one for NAR
        assert torch.all(torch.eq(input_lengths, target_lengths))

        output = self.lm_head(input)
        return output, input_lengths, others
    
    def organize_target(
        self, 
        target: torch.Tensor, 
        target_lengths: torch.Tensor,
        others: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if "layers" not in others:
            raise ValueError("Cannot find the index of the selected layers")

        selected = []
        for idx in others["layers"]:
            selected.append(target[:, :, idx])
        selected = torch.stack(selected, dim=2)

        return selected, target_lengths, others

            
