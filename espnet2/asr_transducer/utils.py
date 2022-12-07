"""Utility functions for Transducer models."""

from typing import List, Tuple, Union

import torch


class TooShortUttError(Exception):
    """Raised when the utt is too short for subsampling.

    Args:
        message: Error message to display.
        actual_size: The size that cannot pass the subsampling.
        limit: The size limit for subsampling.

    """

    def __init__(self, message: str, actual_size: int, limit: int) -> None:
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
        return True, 7
    elif sub_factor == 4 and size < 7:
        return True, 7
    elif sub_factor == 6 and size < 11:
        return True, 11

    return False, -1


def get_convinput_module_parameters(
    input_size: int,
    last_conv_size,
    subsampling_factor: int,
    is_vgg: bool = True,
) -> Tuple[Union[Tuple[int, int], int], int]:
    """Return the convolution module parameters.

    Args:
        input_size: Module input size.
        last_conv_size: Last convolution size for module output size computation.
        subsampling_factor: Total subsampling factor.
        is_vgg: Whether the module type is VGG-like.

    Returns:
        : First MaxPool2D kernel size or second Conv2d kernel size and stride.
        output_size: Convolution module output size.

    """
    if is_vgg:
        # if subsampling_factor not in [4, 6]:
        #     raise ValueError(
        #         "VGG-like input module only support subsampling factor of 4 and 6."
        #     )

        maxpool_kernel1 = subsampling_factor // 2

        output_size = last_conv_size * (((input_size - 1) // 2 - 1) // 2)

        return maxpool_kernel1, output_size

    if subsampling_factor == 2:
        conv_params = (3, 1)
    elif subsampling_factor == 4:
        conv_params = (3, 2)
    else:  # if subsampling_factor == 6:
        conv_params = (5, 3)
    # else:
    #     raise ValueError(
    #         "Conv2D input module only support subsampling factor of 2, 4 and 6."
    #     )

    output_size = last_conv_size * (
        ((input_size - 1) // 2 - (conv_params[0] - conv_params[1])) // conv_params[1]
    )

    return conv_params, output_size


def make_chunk_mask(
    size: int,
    chunk_size: int,
    left_chunk_size: int = 0,
    device: torch.device = None,
) -> torch.Tensor:
    """Create chunk mask for the subsequent steps (size, size).

    Reference: https://github.com/k2-fsa/icefall/blob/master/icefall/utils.py

    Args:
        size: Size of the source mask.
        chunk_size: Number of frames in chunk.
        left_chunk_size: Size of the left context in chunks (0 means full context).
        device: Device for the mask tensor.

    Returns:
        mask: Chunk mask. (size, size)

    """
    mask = torch.zeros(size, size, device=device, dtype=torch.bool)

    for i in range(size):
        if left_chunk_size <= 0:
            start = 0
        else:
            start = max((i // chunk_size - left_chunk_size) * chunk_size, 0)

        end = min((i // chunk_size + 1) * chunk_size, size)
        mask[i, start:end] = True

    return ~mask


def make_source_mask(lengths: torch.Tensor) -> torch.Tensor:
    """Create source mask for given lengths.

    Reference: https://github.com/k2-fsa/icefall/blob/master/icefall/utils.py

    Args:
        lengths: Sequence lengths. (B,)

    Returns:
        : Mask for the sequence lengths. (B, max_len)

    """
    max_len = lengths.max()
    batch_size = lengths.size(0)

    expanded_lengths = torch.arange(max_len).expand(batch_size, max_len).to(lengths)

    return expanded_lengths >= lengths.unsqueeze(1)


def get_transducer_task_io(
    labels: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    ignore_id: int = -1,
    blank_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def pad_list(labels: List[torch.Tensor], padding_value: int = 0):
        """Create padded batch of labels from a list of labels sequences.

        Args:
            labels: Labels sequences. [B x (?)]
            padding_value: Padding value.

        Returns:
            labels: Batch of padded labels sequences. (B,)

        """
        batch_size = len(labels)

        padded = (
            labels[0]
            .new(batch_size, max(x.size(0) for x in labels), *labels[0].size()[1:])
            .fill_(padding_value)
        )

        for i in range(batch_size):
            padded[i, : labels[i].size(0)] = labels[i]

        return padded

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
