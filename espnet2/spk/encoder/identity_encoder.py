# Copyright 2023 Jee-weon Jung
# Apache 2.0

"""RawNet3 Encoder"""

import torch
import torch.nn as nn
from typeguard import check_argument_types

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.spk.layers.rawnet_block import Bottle2neck


class IdentityEncoder(AbsEncoder):
    """
    Identity encoder. Does nothing, just passes frontend feature to the pooling.
    Expected to be used for cases when frontend already has a good
    representation (e.g., SSL features).

    Args:
        input_size: input feature dimension.
    """

    def __init__(
        self,
        input_size: int,
    ):
        super().__init__()
        self._output_size = input_size

    def output_size(self) -> int:
        return self._output_size

    def forward(self, x: torch.Tensor):
        return x.transpose(1, 2)
