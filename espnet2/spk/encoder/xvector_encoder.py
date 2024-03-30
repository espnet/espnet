# x-vector, cross checked with SpeechBrain implementation:
# https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/lobes/models/Xvector.py
# adapted for ESPnet-SPK by Jee-weon Jung
from typing import List

import torch.nn as nn
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder


class XvectorEncoder(AbsEncoder):
    """X-vector encoder. Extracts frame-level x-vector embeddings from features.

    Paper: D. Snyder et al., "X-vectors: Robust dnn embeddings for speaker recognition,"
    in Proc. IEEE ICASSP, 2018.

    Args:
        input_size: input feature dimension.
        ndim: dimensionality of the hidden representation.
        output_size: ouptut embedding dimension.
    """

    @typechecked
    def __init__(
        self,
        input_size: int,
        ndim: int = 512,
        output_size: int = 1500,
        kernel_sizes: List = [5, 3, 3, 1, 1],
        paddings: List = [2, 1, 1, 0, 0],
        dilations: List = [1, 2, 3, 1, 1],
        **kwargs,
    ):
        super().__init__()
        self._output_size = output_size
        in_channels = [input_size] + [ndim] * 4
        out_channels = [ndim] * 4 + [output_size]

        self.layers = nn.ModuleList()
        for idx in range(5):
            self.layers.append(
                nn.Conv1d(
                    in_channels[idx],
                    out_channels[idx],
                    kernel_sizes[idx],
                    dilation=dilations[idx],
                    padding=paddings[idx],
                )
            )
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(out_channels[idx]))

    def output_size(self) -> int:
        return self._output_size

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, S, D) -> (B, D, S)
        for layer in self.layers:
            x = layer(x)

        return x
