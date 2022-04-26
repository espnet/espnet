"""Utility functions for Transducer models."""

from typing import Tuple

import torch

from espnet.nets.pytorch_backend.nets_utils import pad_list


class TooShortUttError(Exception):
    """Raised when the utt is too short for subsampling.

    Args:
        message: Error message to display.
        actual_size: The size that cannot pass the subsampling.
        limit: The size limit for subsampling.

    """

    def __init__(self, message: str, actual_size: int, limit: int):
        """Construct a TooShortUttError module."""
        super().__init__(message)

        self.actual_size = actual_size
        self.limit = limit


def check_short_utt(sub_factor: int, size: int) -> Tuple[bool, int]:
    """Check if the input is too short for subsampling.

    Args:
        sub_factor: Subsampling factor for Conv2DSubsampling.
        size: Input size.

    Returns:
        : Whether an error should be sent.
        : Size limit for specified subsampling factor.

    """
    if sub_factor == 2 and size < 3:
        return True, 3
    elif sub_factor == 4 and size < 7:
        return True, 7
    elif sub_factor == 6 and size < 11:
        return True, 11

    return False, -1


def sub_factor_to_params(sub_factor: int, dim_input: int) -> Tuple[int, int, int]:
    """Get conv2D second layer parameters for given subsampling factor.

    Args:
        sub_factor: Subsampling factor (1/X).
        dim_input: Input dimension.

    Returns:
        : Kernel size for second convolution.
        : Stride for second convolution.
        : Output dimension for Conv2DSubsampling module.

    """
    if sub_factor == 2:
        return 3, 1, (((dim_input - 1) // 2 - 2))
    elif sub_factor == 4:
        return 3, 2, (((dim_input - 1) // 2 - 1) // 2)
    elif sub_factor == 6:
        return 5, 3, (((dim_input - 1) // 2 - 2) // 3)
    else:
        raise ValueError(
            "subsampling_factor parameter should be set to either 2, 4 or 6."
        )


def get_transducer_task_io(
    labels: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    ignore_id: int = -1,
    blank_id: int = 0,
):
    """Get Transducer loss I/O.

    Args:
        labels: Label ID sequences. (B, L)
        encoder_out_lens: Encoder output lengths. (B,)
        ignore_id: Padding symbol ID.
        blank_id: Blank symbol ID.

    Returns:
        decoder_in: Decoder inputs. (B, U)
        target: Target label ID sequences. (B, U)
        t_len: Time lengths. (B,)
        u_len: Label lengths. (B,)

    """
    device = labels.device

    labels_unpad = [y[y != ignore_id] for y in labels]
    blank = labels[0].new([blank_id])

    decoder_in = pad_list(
        [torch.cat([blank, label], dim=0) for label in labels_unpad], blank_id
    ).to(device)

    target = pad_list(labels_unpad, blank_id).type(torch.int32).to(device)

    encoder_out_lens = list(map(int, encoder_out_lens))
    t_len = torch.IntTensor(encoder_out_lens).to(device)

    u_len = torch.IntTensor([y.size(0) for y in labels_unpad]).to(device)

    return decoder_in, target, t_len, u_len
