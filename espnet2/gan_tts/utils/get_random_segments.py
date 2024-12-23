# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Function to get random segments."""

from typing import Tuple

import torch


def get_random_segments(
    x: torch.Tensor,
    x_lengths: torch.Tensor,
    segment_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Function to get random segments from an input tensor.

    This function extracts random segments of a specified size from the input tensor,
    ensuring that the segments do not exceed the lengths provided in the `x_lengths` tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, T), where B is the batch size,
            C is the number of channels, and T is the length of the input.
        x_lengths (torch.Tensor): Length tensor of shape (B,), indicating the valid
            lengths of each input tensor in the batch.
        segment_size (int): Size of the segment to be extracted from the input tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - Tensor: Segmented tensor of shape (B, C, segment_size).
            - Tensor: Start index tensor of shape (B,), indicating the starting
              indices of the segments in the input tensor.

    Examples:
        >>> x = torch.randn(4, 2, 10)  # Example input tensor (B=4, C=2, T=10)
        >>> x_lengths = torch.tensor([10, 9, 8, 7])  # Valid lengths for each input
        >>> segment_size = 5
        >>> segments, start_idxs = get_random_segments(x, x_lengths, segment_size)
        >>> segments.shape
        torch.Size([4, 2, 5])
        >>> start_idxs.shape
        torch.Size([4])
    """
    batches = x.shape[0]
    max_start_idx = x_lengths - segment_size
    max_start_idx[max_start_idx < 0] = 0
    start_idxs = (torch.rand([batches]).to(x.device) * max_start_idx).to(
        dtype=torch.long,
    )
    segments = get_segments(x, start_idxs, segment_size)

    return segments, start_idxs


def get_segments(
    x: torch.Tensor,
    start_idxs: torch.Tensor,
    segment_size: int,
) -> torch.Tensor:
    """
        Function to get random segments.

    This function retrieves random segments from a given tensor based on specified
    segment sizes and their respective lengths. The output consists of both the
    segmented tensor and the starting indices of these segments.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, T) where:
            - B: Batch size
            - C: Number of channels
            - T: Length of the sequence
        x_lengths (torch.Tensor): Length tensor of shape (B,) indicating the valid
            lengths of each input tensor in the batch.
        segment_size (int): The size of the segments to be extracted from the
            input tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - Tensor: Segmented tensor of shape (B, C, segment_size).
            - Tensor: Start index tensor of shape (B,) representing the starting
              indices of each segment.

    Examples:
        >>> x = torch.randn(4, 2, 10)  # A batch of 4 samples, 2 channels, 10 length
        >>> x_lengths = torch.tensor([10, 8, 10, 5])  # Lengths of each sample
        >>> segment_size = 5
        >>> segments, start_idxs = get_random_segments(x, x_lengths, segment_size)
        >>> segments.shape
        torch.Size([4, 2, 5])  # Segmented tensor shape
        >>> start_idxs.shape
        torch.Size([4])  # Start index tensor shape
    """
    b, c, _ = x.size()
    segments = x.new_zeros(b, c, segment_size)
    for i, start_idx in enumerate(start_idxs):
        segments[i] = x[i, :, start_idx : start_idx + segment_size]
    return segments
