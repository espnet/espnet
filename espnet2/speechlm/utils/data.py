# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Data utilities for tensor manipulation and device management in SpeechLM."""

import dataclasses
from typing import List, Tuple, Union

import numpy as np
import torch


def pad_list(
    sequences: List[Union[np.ndarray, torch.Tensor]], pad_value: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of sequences to the same length and stack them.

    Uses right padding (padding at the end of sequences).
    Assumes the FIRST dimension is the time/sequence length.

    Args:
        sequences: List of sequences to pad and stack.
                  Each can be of any shape [seq_len, ...] where seq_len is variable
        pad_value: Value to use for padding (default: 0.0)

    Returns:
        Tuple of:
        - Padded and stacked tensor of shape [batch, max_seq_len, ...]
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
        seq_len = t.shape[0]  # First dimension is the length
        lengths.append(seq_len)
        max_len = max(max_len, seq_len)

    # Validate shapes and find common dtype
    other_dims = tensors[0].shape[1:]  # All dimensions except the first

    for t in tensors[1:]:
        if t.shape[1:] != other_dims:
            raise ValueError(
                "All sequences must have the same shape except for the first dimension"
            )

    # Find common dtype using torch's type promotion
    dtype = tensors[0].dtype
    for t in tensors[1:]:
        dtype = torch.promote_types(dtype, t.dtype)

    # Pre-allocate output tensor
    batch_size = len(tensors)
    output_shape = (batch_size, max_len) + other_dims
    padded = torch.full(output_shape, pad_value, dtype=dtype, device=device)

    # Fill sequences
    for i, t in enumerate(tensors):
        padded[i, : lengths[i], ...] = t.to(
            dtype=dtype, device=device, non_blocking=True
        )

    # Create length tensor
    length_tensor = torch.tensor(lengths, dtype=torch.long, device=device)

    return padded, length_tensor


# NOTE(Jinchuan): copy from the existing code:
# espnet2.torch_utils.device_funcs: to_device
def to_device(data, device=None, dtype=None, non_blocking=False, copy=False):
    """Change the device of object recursively"""
    if isinstance(data, dict):
        return {
            k: to_device(v, device, dtype, non_blocking, copy) for k, v in data.items()
        }
    elif dataclasses.is_dataclass(data) and not isinstance(data, type):
        return type(data)(
            *[
                to_device(v, device, dtype, non_blocking, copy)
                for v in dataclasses.astuple(data)
            ]
        )
    # maybe namedtuple. I don't know the correct way to judge namedtuple.
    elif isinstance(data, tuple) and type(data) is not tuple:
        return type(data)(
            *[to_device(o, device, dtype, non_blocking, copy) for o in data]
        )
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(v, device, dtype, non_blocking, copy) for v in data)
    elif isinstance(data, np.ndarray):
        return to_device(torch.from_numpy(data), device, dtype, non_blocking, copy)
    elif isinstance(data, torch.Tensor):
        if dtype is not None:
            dtype = str(dtype).removeprefix("torch.")
            cur_dtype = str(data.dtype).removeprefix("torch.")

            if not (
                ("int" in dtype and "int" in cur_dtype)
                or ("float" in dtype and "float" in cur_dtype)
            ):
                dtype = None  # avoid conversion between int and float.
            else:
                dtype = getattr(torch, dtype)

        return data.to(device, dtype, non_blocking, copy)
    else:
        return data
