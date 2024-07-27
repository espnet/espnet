#!/usr/bin/env python3
#  2020, Technische Universität München;  Ludwig Kürzinger
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Sliding Window for raw audio input data."""

from typing import Optional, Tuple

import torch
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend


class SlidingWindow(AbsFrontend):
    """
        Sliding Window for raw audio input data.

    This class implements a sliding window operation over batched continuous raw audio tensors.
    It is designed to be used as a frontend in audio processing pipelines, particularly
    in combination with pre-encoders compatible with raw audio data (e.g., Sinc convolutions).

    The sliding window operation segments the input audio into overlapping frames,
    which can be further processed by subsequent layers in the neural network.

    Attributes:
        win_length (int): Length of each window frame.
        hop_length (int): Number of samples to advance between successive frames.
        channels (int): Number of input audio channels.
        padding (Optional[int]): Padding option (currently not implemented).
        fs (Optional[int]): Sampling rate (placeholder for compatibility, not used).

    Note:
        - Output length is calculated incorrectly if audio is shorter than win_length.
        - Trailing values are currently discarded as padding is not yet implemented.
        - No additional window function is applied to input values.

    Examples:
        >>> sliding_window = SlidingWindow(win_length=400, hop_length=160, channels=1)
        >>> input_tensor = torch.randn(32, 16000, 1)  # (batch_size, time_steps, channels)
        >>> input_lengths = torch.full((32,), 16000)
        >>> output, output_lengths = sliding_window(input_tensor, input_lengths)
        >>> print(output.shape)
        torch.Size([32, 98, 1, 400])
    """

    @typechecked
    def __init__(
        self,
        win_length: int = 400,
        hop_length: int = 160,
        channels: int = 1,
        padding: Optional[int] = None,
        fs=None,
    ):
        """Initialize.

        Args:
            win_length: Length of frame.
            hop_length: Relative starting point of next frame.
            channels: Number of input channels.
            padding: Padding (placeholder, currently not implemented).
            fs:  Sampling rate (placeholder for compatibility, not used).
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
        """
                Apply a sliding window on the input.

        This method processes the input audio tensor by applying a sliding window,
        segmenting the continuous audio into overlapping frames.

        Args:
            input (torch.Tensor): Input tensor of shape (B, T, C*D) or (B, T*C*D),
                where B is batch size, T is time steps, C is channels, and D is 1.
            input_lengths (torch.Tensor): Tensor containing the valid length of each
                sample in the batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - output (torch.Tensor): Windowed output of shape (B, T, C, D),
                  where D is the window length (win_length).
                - output_lengths (torch.Tensor): Tensor containing the valid length
                  of each windowed sample in the batch.

        Examples:
            >>> sliding_window = SlidingWindow(win_length=400, hop_length=160, channels=1)
            >>> input_tensor = torch.randn(32, 16000, 1)  # (batch_size, time_steps, channels)
            >>> input_lengths = torch.full((32,), 16000)
            >>> output, output_lengths = sliding_window.forward(input_tensor, input_lengths)
            >>> print(output.shape)
            torch.Size([32, 98, 1, 400])
            >>> print(output_lengths)
            tensor([98, 98, 98, ..., 98, 98, 98])

        Note:
            The output length is calculated as:
            (input_length - win_length) // hop_length + 1
        """
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
        output_lengths = (
            torch.div(
                input_lengths - self.win_length, self.hop_length, rounding_mode="trunc"
            )
            + 1
        )
        return output, output_lengths

    def output_size(self) -> int:
        """
                Return the output size of the feature dimension.

        This method returns the length of the window, which corresponds to the
        size of the feature dimension (D) in the output tensor.

        Returns:
            int: The window length (win_length).

        Examples:
            >>> sliding_window = SlidingWindow(win_length=400, hop_length=160, channels=1)
            >>> print(sliding_window.output_size())
            400

        Note:
            This value is used to determine the size of the last dimension in the
            output tensor returned by the forward method.
        """
        return self.win_length
