from typing import Tuple, Dict

import torch

from espnet2.speechlm.core_lm.abs_core_lm import AbsCoreLM


class BuiltinCoreLM(AbsCoreLM):

    def forward(
        self, input: torch.Tensor, input_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        raise NotImplementedError

    def inference(
        self, prefix: torch.Tensor, input_mask: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        raise NotImplementedError
