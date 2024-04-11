from abc import ABC, abstractmethod
from typing import Tuple, Dict

import torch

from espnet2.speechlm.predictor.abs_predictor import AbsPredictor


class ParallelPredcitor(AbsPredictor):
    def forward(
        self, input: torch.Tensor, input_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        raise NotImplementedError

    def inference(
        self, prefix: torch.Tensor, input_mask: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        raise NotImplementedError
