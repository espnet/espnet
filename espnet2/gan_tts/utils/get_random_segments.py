# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Function to get random segments."""

from typing import Optional, Tuple

import torch


def get_random_segments(
    x: torch.Tensor,
    x_lengths: torch.Tensor,
    segment_size: int,
    pitch: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get random segments.

    Args:
        x (Tensor): Input tensor (B, C, T).
        x_lengths (Tensor): Length tensor (B,).
        segment_size (int): Segment size.

    Returns:
        Tensor: Segmented tensor (B, C, segment_size).
        Tensor: Start index tensor (B,).

    """
    b, c, t = x.size()
    max_start_idx = x_lengths - segment_size
    start_idxs = (torch.rand([b]).to(x.device) * max_start_idx).to(
        dtype=torch.long,
    )
    segments = get_segments(x, start_idxs, segment_size)
    if pitch is not None:
        pitch_segments = get_segments(pitch, start_idxs, segment_size)
    else:
        pitch_segments = None
    return segments, start_idxs, pitch_segments


def get_segments(
    x: torch.Tensor,
    start_idxs: torch.Tensor,
    segment_size: int,
) -> torch.Tensor:
    """Get segments.

    Args:
        x (Tensor): Input tensor (B, C, T).
        start_idxs (Tensor): Start index tensor (B,).
        segment_size (int): Segment size.

    Returns:
        Tensor: Segmented tensor (B, C, segment_size).

    """
    b, c, t = x.size()
    segments = x.new_zeros(b, c, segment_size)
    for i, start_idx in enumerate(start_idxs):
        segments[i] = x[i, :, start_idx : start_idx + segment_size]
    return segments
