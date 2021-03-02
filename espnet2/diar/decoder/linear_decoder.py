#!/usr/bin/env python3

# Copyright 2021 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch

from espnet2.diar.decoder.abs_decoder import AbsDecoder


class LinearDecoder(AbsDecoder):
    """Linear decoder for speaker diarization """

    def __init__(
        self,
        encoder_output_size: int,
        num_spk: int = 2,
    ):
        super().__init__()
        self.linear_decoder = torch.nn.Linear(encoder_output_size, output_size)

    def forward(self, input: torch.Tensor, ilens: torch.Tensor):
        """Forward.
        Args:
            input (torch.Tensor): hidden_space [Batch, T, F]
            ilens (torch.Tensor): input lengths [Batch]
        """

        output = self.linear_decoder(input)

        return output

    @property
    def num_spk(self):
        return self._num_spk
