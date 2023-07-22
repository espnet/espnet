"""Retention module with parallel and recurrent modes."""

import math
from typing import Dict, Optional, Tuple

import torch

from espnet2.asr_transducer.encoder.modules.rotary_embedding import (  # noqa: H301
    RotaryPositionalEmbedding,
)


class Retention(torch.nn.Module):
    """Retention module definition.

    Args:
        size: Hidden size.
        num_heads: Number of heads.
        decay_length: Maximum length for the decaying mask.
        dropout_rate: Dropout rate.

    """

    def __init__(
        self,
        size: int,
        num_heads: int,
        decay_length: int = 768,
        dropout_rate: float = 0.0,
    ) -> None:
        """Construct a Retention object."""
        super().__init__()

        d_qkv = size // num_heads

        self.proj_qkv = torch.nn.Linear(size, 3 * size)
        self.rope = RotaryPositionalEmbedding(d_qkv)

        self.dropout = torch.nn.Dropout(p=dropout_rate)

        self.size = size
        self.d_qkv = d_qkv
        self.num_heads = num_heads

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(decay_length, decay_length)).view(
                1, 1, decay_length, decay_length
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform parallel retention.

        Args:
            x: Retention input sequences. (B, T, D_enc)

        Returns:
            x: Retention output sequences. (B, T, D_enc)

        """
        batch, length, dim = x.size()

        query, key, value = self.proj_qkv(x).split(self.size, dim=2)

        query = query.view(batch, length, self.num_heads, self.d_qkv).transpose(1, 2)
        key = key.view(batch, length, self.num_heads, self.d_qkv).transpose(1, 2)
        value = value.view(batch, length, self.num_heads, self.d_qkv).transpose(1, 2)

        query, key = self.rope(query, key)
        decay_mask = self.mask[:, :, :length, :length]

        retention = query @ key.transpose(-1, -2)
        retention = self.dropout(retention * decay_mask)

        x = retention @ value
        x = x.transpose(1, 2).contiguous().view(batch, length, dim)

        return x

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """Perform recurrent retention.

        Args:
            x: Retention input sequences. (B, 1, D_enc)

        Returns:
            x: Retention output sequences. (B, 1, D_enc)

        """
        raise NotImplementedError
