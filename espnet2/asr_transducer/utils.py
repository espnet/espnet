"""Utility functions for Transducer models."""

from typing import List, Tuple, Union

import torch


class TooShortUttError(Exception):
    """
    Raised when the utterance is too short for subsampling.

    This exception is thrown to indicate that the size of the input
    utterance does not meet the minimum requirement for the specified
    subsampling factor.

    Attributes:
        actual_size (int): The size of the input that failed the subsampling check.
        limit (int): The minimum size limit required for the subsampling.

    Args:
        message (str): Error message to display.
        actual_size (int): The size that cannot pass the subsampling.
        limit (int): The size limit for subsampling.

    Examples:
        >>> raise TooShortUttError("Input size too short", 2, 3)
        Traceback (most recent call last):
            ...
        TooShortUttError: Input size too short
    """

    def __init__(self, message: str, actual_size: int, limit: int) -> None:
        """Construct a TooShortUttError module."""
        super().__init__(message)

        self.actual_size = actual_size
        self.limit = limit


def check_short_utt(sub_factor: int, size: int) -> Tuple[bool, int]:
    """
    Check if the input is too short for subsampling.

    This function evaluates whether the input size is adequate for the given
    subsampling factor. If the input size is shorter than the required length
    for the specified subsampling factor, it returns an error indication along
    with the minimum size limit for that factor.

    Args:
        sub_factor: Subsampling factor for Conv2DSubsampling.
        size: Input size.

    Returns:
        Tuple[bool, int]: A tuple where the first element indicates whether an
        error should be sent (True if the size is too short, otherwise False),
        and the second element is the size limit for the specified subsampling
        factor. A return value of -1 indicates no limit.

    Examples:
        >>> check_short_utt(2, 2)
        (True, 7)
        
        >>> check_short_utt(4, 6)
        (True, 7)
        
        >>> check_short_utt(6, 12)
        (False, -1)

    Note:
        The size limits for each subsampling factor are as follows:
        - For sub_factor 2, size must be at least 3.
        - For sub_factor 4, size must be at least 7.
        - For sub_factor 6, size must be at least 11.
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
    """
    Return the convolution module parameters.

    This function calculates the parameters for a convolutional module
    based on the input size, last convolution size, subsampling factor,
    and whether the architecture follows a VGG-like structure.

    Args:
        input_size: Module input size.
        last_conv_size: Last convolution size for module output size 
            computation.
        subsampling_factor: Total subsampling factor.
        is_vgg: Whether the module type is VGG-like. Defaults to True.

    Returns:
        A tuple containing:
            - First MaxPool2D kernel size or second Conv2D kernel size 
              and stride.
            - output_size: Convolution module output size.

    Examples:
        >>> get_convinput_module_parameters(64, 32, 2, True)
        (1, 15)

        >>> get_convinput_module_parameters(64, 32, 4, False)
        ((3, 2), 12)

    Note:
        The output size is computed based on the input size and the 
        specified subsampling factor. For VGG-like architectures, 
        the calculation may differ from standard convolutional networks.
    """
    if is_vgg:
        maxpool_kernel1 = subsampling_factor // 2

        output_size = last_conv_size * (((input_size - 1) // 2 - 1) // 2)

        return maxpool_kernel1, output_size

    if subsampling_factor == 2:
        conv_params = (3, 1)
    elif subsampling_factor == 4:
        conv_params = (3, 2)
    else:
        conv_params = (5, 3)

    output_size = last_conv_size * (
        ((input_size - 1) // 2 - (conv_params[0] - conv_params[1])) // conv_params[1]
    )

    return conv_params, output_size


def make_chunk_mask(
    size: int,
    chunk_size: int,
    num_left_chunks: int = 0,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Create a chunk mask for the subsequent steps.

    This function generates a boolean mask tensor indicating which 
    frames can be attended to based on the chunking strategy defined 
    by the `chunk_size` and `num_left_chunks`. The resulting mask 
    tensor has a shape of (size, size), where `size` is the total 
    number of frames.

    Reference:
        https://github.com/k2-fsa/icefall/blob/master/icefall/utils.py

    Args:
        size: Size of the source mask.
        chunk_size: Number of frames in each chunk.
        num_left_chunks: Number of left chunks that the attention 
                         module can see. A null or negative value 
                         means full context.
        device: The device for the mask tensor (e.g., 'cpu' or 'cuda').

    Returns:
        mask: A boolean tensor of shape (size, size) representing 
              the chunk mask, where `True` indicates frames that can 
              be attended to, and `False` indicates masked frames.

    Examples:
        >>> mask = make_chunk_mask(size=10, chunk_size=3, num_left_chunks=1)
        >>> print(mask)
        tensor([[False,  True,  True, False, False, False, False, False, False, False],
                [ True,  True,  True, False, False, False, False, False, False, False],
                [False,  True,  True, False, False, False, False, False, False, False],
                [False, False, False,  True,  True,  True, False, False, False, False],
                [False, False, False,  True,  True,  True, False, False, False, False],
                ...
               ])
    """
    mask = torch.zeros(size, size, device=device, dtype=torch.bool)

    for i in range(size):
        if num_left_chunks <= 0:
            start = 0
        else:
            start = max((i // chunk_size - num_left_chunks) * chunk_size, 0)

        end = min((i // chunk_size + 1) * chunk_size, size)
        mask[i, start:end] = True

    return ~mask


def make_source_mask(lengths: torch.Tensor) -> torch.Tensor:
    """
    Create source mask for given lengths.

    This function generates a source mask for a batch of sequences based on their
    lengths. The mask is a binary tensor where each position is marked as True if it
    is valid (i.e., within the length of the corresponding sequence) and False otherwise.

    Reference:
        https://github.com/k2-fsa/icefall/blob/master/icefall/utils.py

    Args:
        lengths: Sequence lengths. (B,)

    Returns:
        torch.Tensor: Mask for the sequence lengths. (B, max_len)

    Examples:
        >>> lengths = torch.tensor([3, 5, 2])
        >>> mask = make_source_mask(lengths)
        >>> print(mask)
        tensor([[ True,  True,  True, False, False],
                [ True,  True,  True,  True,  True],
                [ True,  True, False, False, False]])
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
    """
    Get Transducer loss I/O.

    This function prepares the input and target sequences required for 
    calculating the Transducer loss. It processes the provided label 
    sequences and encoder output lengths, handling padding and 
    blank symbol insertion as necessary.

    Args:
        labels: Label ID sequences. Shape: (B, L) where B is the batch size 
                and L is the maximum label length.
        encoder_out_lens: Encoder output lengths. Shape: (B,) indicating 
                           the length of the encoder output for each sequence.
        ignore_id: Padding symbol ID, which will be ignored in the labels. 
                   Default is -1.
        blank_id: Blank symbol ID, which is prepended to the decoder input. 
                  Default is 0.

    Returns:
        decoder_in: Decoder inputs. Shape: (B, U) where U is the maximum 
                     number of tokens after prepending the blank symbol.
        target: Target label ID sequences. Shape: (B, U) where U is the 
                number of valid tokens in each batch after ignoring 
                the padding symbols.
        t_len: Time lengths of the encoder outputs. Shape: (B,) where each 
               entry corresponds to the length of the encoder output for 
               the respective input sequence.
        u_len: Lengths of the target label sequences. Shape: (B,) where each 
               entry corresponds to the number of valid tokens in the target 
               sequence.

    Examples:
        >>> labels = torch.tensor([[1, 2, 3, -1], [1, -1, -1, -1]])
        >>> encoder_out_lens = torch.tensor([4, 1])
        >>> decoder_in, target, t_len, u_len = get_transducer_task_io(labels, 
        ... encoder_out_lens)
        >>> print(decoder_in)
        tensor([[0, 1, 2, 3],
                [0, 1]])
        >>> print(target)
        tensor([[1, 2, 3],
                [1]])
        >>> print(t_len)
        tensor([4, 1])
        >>> print(u_len)
        tensor([3, 1])

    Note:
        The function assumes that the input tensors are on the same device 
        (CPU or GPU).

    Raises:
        ValueError: If the input tensors have incompatible shapes or if 
                    there are invalid IDs in the label sequences.
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
