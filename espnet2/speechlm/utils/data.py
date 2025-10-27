"""Data utilities for SpeechLM module."""

from typing import List, Tuple, Union

import numpy as np
import torch


def pad_list(
    sequences: List[Union[np.ndarray, torch.Tensor]], pad_value: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of sequences to the same length and stack them.

    Uses right padding (padding at the end of sequences).
    Assumes the LAST dimension is the time/sequence length.

    Args:
        sequences: List of sequences to pad and stack.
                  Each can be of any shape [..., seq_len] where seq_len is variable
        pad_value: Value to use for padding (default: 0.0)

    Returns:
        Tuple of:
        - Padded and stacked tensor of shape [batch, ..., max_seq_len]
        - Length tensor of shape [batch] with original sequence lengths (dtype=long)

    Raises:
        ValueError: If sequence list is empty or sequences have inconsistent shapes
        TypeError: If sequences contain non-tensor/non-array types
    """
    if not sequences:
        raise ValueError("Empty sequence list")

    # Convert to tensors and collect info in one pass
    tensors = []
    lengths = []
    max_len = 0
    device = None

    for seq in sequences:
        if isinstance(seq, np.ndarray):
            t = torch.from_numpy(seq)
        elif isinstance(seq, torch.Tensor):
            t = seq
            if device is None:
                device = t.device
        else:
            raise TypeError(f"Expected numpy array or torch tensor, got {type(seq)}")

        tensors.append(t)
        seq_len = t.shape[-1]
        lengths.append(seq_len)
        max_len = max(max_len, seq_len)

    # Validate shapes and find common dtype
    other_dims = tensors[0].shape[:-1]

    for t in tensors[1:]:
        if t.shape[:-1] != other_dims:
            raise ValueError(
                "All sequences must have the same shape except for the last dimension"
            )

    # Find common dtype using torch's type promotion
    dtype = tensors[0].dtype
    for t in tensors[1:]:
        dtype = torch.promote_types(dtype, t.dtype)

    # Pre-allocate output tensor
    batch_size = len(tensors)
    output_shape = (batch_size,) + other_dims + (max_len,)
    padded = torch.full(output_shape, pad_value, dtype=dtype, device=device)

    # Fill sequences
    for i, t in enumerate(tensors):
        padded[i, ..., : lengths[i]] = t.to(
            dtype=dtype, device=device, non_blocking=True
        )

    # Create length tensor
    length_tensor = torch.tensor(lengths, dtype=torch.long, device=device)

    return padded, length_tensor
