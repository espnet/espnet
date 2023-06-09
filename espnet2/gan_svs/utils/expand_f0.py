# Copyright 2023 Yifeng Yu
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Function to get random segments."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def expand_f0(f0_frame, hop_length, method="interpolation"):
    """Expand f0 to output wave length.

    Args:
        f0_frame (Tensor): Input tensor (B, 1, frame_len).
        hop_length (Tensor): Hop length.
        method (str): Method to expand f0. Choose either 'interpolation' or 'repeat'.

    Returns:
        Tensor: Output tensor (B, 1, wav_len).

    """
    frame_length = f0_frame.size(2)
    signal_length = frame_length * hop_length
    if method == "interpolation":
        f0_sample = F.interpolate(
            f0_frame, size=signal_length, mode="linear", align_corners=False
        )
    elif method == "repeat":
        f0_sample = f0_frame.repeat_interleave(hop_length, dim=2)[:signal_length]
    else:
        raise ValueError("Invalid method. Choose either 'interpolation' or 'repeat'.")
    f0_sample = f0_sample.squeeze()[
        :signal_length
    ]  # Remove extra dimensions and trim to signal_length
    return f0_sample
