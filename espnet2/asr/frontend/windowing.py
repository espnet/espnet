#!/usr/bin/env python3
# encoding: utf-8
#  2020, Technische Universität München;  Ludwig Kürzinger
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Sliding Window for raw audio input data."""

from espnet2.asr.frontend.abs_frontend import AbsFrontend
import torch
from typing import Tuple


class SlidingWindow(AbsFrontend):
    """Sliding Window.

    Provides a sliding window over a batched continuous raw audio tensor.
    Optionally, provides padding. (Currently not implemented)

    Known issues:
    Output length is calculated incorrectly if audio shorter than win_length.
    WARNING: trailing values are discarded - padding not implemented yet.
    There is currently no additional window function applied to input values.

    input: Tensor with (B, T, C*D) or (B, T*C*D), With D=1
    output: Tensor with dimensions (B, T, C, D)
    """

    def __init__(
        self,
        win_length=400,
        hop_length=160,
        channels=1,
        padding=None,
        fs=None,
    ):
        """Initialize.

        :param win_length: Length of frame.
        :param hop_length: Relative starting point of next frame.
        :param channels: Number of input channels.
        :param padding: Padding (placeholder, currently not implemented).
        :param fs: Sampling rate (placeholder for compatibility, not used).
        """
        super().__init__()
        self.fs = fs
        self.win_length = win_length
        self.hop_length = hop_length
        self.channels = channels
        self.padding = padding

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward."""
        input_size = input.size()
        B = input_size[0]
        T = input_size[1]
        C = self.channels
        D = self.win_length
        # (B, T, C) --> (T, B, C)
        continuous = input.view(B, T, C).permute(1, 0, 2)
        windowed = continuous.unfold(0, D, self.hop_length)
        # (T, B, C, D) --> (B, T, C, D)
        output = windowed.permute(1, 0, 2, 3).contiguous()
        # After unfold(), windowed lengths change:
        output_lengths = (input_lengths - self.win_length) // self.hop_length + 1
        return output, output_lengths

    def output_size(self) -> int:
        """Return output length of feature dimension D, i.e. the window length."""
        return self.win_length
