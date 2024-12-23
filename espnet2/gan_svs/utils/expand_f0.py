# Copyright 2023 Yifeng Yu
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Function to get random segments."""

import torch.nn.functional as F


def expand_f0(f0_frame, hop_length, method="interpolation"):
    """
        Expand the fundamental frequency (f0) to match the output waveform length.

    This function takes an input tensor representing f0 values across frames and expands
    it to a longer output tensor corresponding to the desired waveform length using
    specified methods.

    Attributes:
        f0_frame (Tensor): Input tensor with shape (B, 1, frame_len) representing f0
                           values for each frame.
        hop_length (Tensor): The hop length to calculate the output waveform length.
        method (str): The method used for expansion, either 'interpolation' or 'repeat'.

    Args:
        f0_frame (Tensor): Input tensor (B, 1, frame_len) containing f0 values.
        hop_length (int): The hop length to determine the output length.
        method (str): The method to expand f0. Choose either 'interpolation' or
                      'repeat'.

    Returns:
        Tensor: Output tensor (B, 1, wav_len) containing expanded f0 values.

    Raises:
        ValueError: If the specified method is not 'interpolation' or 'repeat'.

    Examples:
        >>> import torch
        >>> f0 = torch.randn(2, 1, 5)  # Example f0 frame tensor
        >>> hop_length = 4
        >>> expanded_f0 = expand_f0(f0, hop_length, method="interpolation")
        >>> expanded_f0.shape
        torch.Size([2, 1, 20])

        >>> expanded_f0_repeat = expand_f0(f0, hop_length, method="repeat")
        >>> expanded_f0_repeat.shape
        torch.Size([2, 1, 20])

    Note:
        The output tensor length is determined by the product of hop_length and the
        input frame length.

    Todo:
        - Add more methods for f0 expansion in the future if needed.
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
