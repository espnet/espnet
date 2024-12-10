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
    Sliding Window.

    This class provides a sliding window mechanism over a batched continuous raw 
    audio tensor. It is designed to be used in conjunction with a pre-encoder 
    compatible with raw audio data, such as Sinc convolutions. The class currently 
    does not implement padding, and there are known issues regarding output length 
    calculation when the audio input is shorter than the specified window length. 
    Please note that trailing values are discarded due to the lack of padding.

    Attributes:
        fs (Optional): Sampling rate (not used currently).
        win_length (int): Length of the frame for the sliding window.
        hop_length (int): Relative starting point of the next frame.
        channels (int): Number of input channels.
        padding (Optional[int]): Placeholder for padding (not implemented).

    Args:
        win_length (int): Length of frame (default: 400).
        hop_length (int): Relative starting point of next frame (default: 160).
        channels (int): Number of input channels (default: 1).
        padding (Optional[int]): Padding (currently not implemented).
        fs: Sampling rate (placeholder for compatibility, not used).

    Known Issues:
        - Output length is calculated incorrectly if audio is shorter than 
        win_length.
        - WARNING: trailing values are discarded - padding not implemented yet.
        - No additional window function is applied to input values.

    Examples:
        >>> sliding_window = SlidingWindow(win_length=400, hop_length=160)
        >>> input_tensor = torch.randn(2, 800, 1)  # Example input
        >>> input_lengths = torch.tensor([800, 800])  # Example lengths
        >>> output, output_lengths = sliding_window.forward(input_tensor, input_lengths)
        >>> print(output.shape)  # Should show the output shape
        >>> print(output_lengths)  # Should show the output lengths
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
        Apply a sliding window on the input tensor.

        This method processes a batch of audio input data using a sliding window
        approach, which allows the model to handle continuous audio signals in 
        manageable frames. The method outputs the windowed audio along with 
        the corresponding lengths of the output sequences.

        Args:
            input: A tensor of shape (B, T, C*D) or (B, T*C*D), where:
                - B is the batch size,
                - T is the length of the input sequence,
                - C is the number of input channels,
                - D is the window length (for the case of (B, T*C*D), 
                  it is assumed that D=1).
            input_lengths: A tensor of shape (B,) representing the lengths 
                of each input sequence in the batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - A tensor of shape (B, T, C, D) representing the 
                  windowed output, where D is the window length.
                - A tensor of shape (B,) representing the output lengths 
                  for each sequence in the batch.

        Examples:
            >>> import torch
            >>> sliding_window = SlidingWindow(win_length=400, hop_length=160)
            >>> input_tensor = torch.randn(2, 800, 1)  # (B=2, T=800, C=1)
            >>> input_lengths = torch.tensor([800, 800])  # Lengths for each batch
            >>> output, output_lengths = sliding_window.forward(input_tensor, input_lengths)
            >>> print(output.shape)  # Should output (2, num_windows, 1, 400)
            >>> print(output_lengths)  # Output lengths based on input lengths

        Note:
            - The method currently does not apply any window function to 
              the input values.
            - Trailing values may be discarded due to the absence of padding 
              implementation.

        Raises:
            ValueError: If the input tensor dimensions do not match the 
                expected shape.
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
        Return the output length of the feature dimension D.

        This method provides the length of the output feature dimension D, 
        which corresponds to the defined window length used in the sliding 
        window operation.

        Returns:
            int: The length of the output feature dimension D, which is equal 
            to the window length (win_length).

        Examples:
            >>> sliding_window = SlidingWindow(win_length=400)
            >>> sliding_window.output_size()
            400
        """
        return self.win_length
