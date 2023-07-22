"""Rotary positional embedding module."""

import math
from typing import Optional, Tuple

import torch


class RotaryPositionalEmbedding(torch.nn.Module):
    """RotaryPositionalEmbedding module definition.

    Args:
        size: Embedding size.

    """

    def __init__(self, size: int) -> None:
        """Construct a RotaryPositionalEmbedding object."""
        super().__init__()

        inv_freq = 1.0 / (10000 ** (torch.arange(0, size, 2).float() / size))

        self.size = size

        self.max_len = None
        self.cached_cos = None
        self.cached_sin = None

        self.register_buffer("inv_freq", inv_freq)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the dimension of the inputs.

        Args:
            x: Input sequence. (B, num_heads, length, size)

        Returns:
            x: Rotated output sequence. (B, num_heads, length, size)

        """
        x1, x2 = x.chunk(2, dim=-1)

        x = torch.cat((-x2, x1), dim=-1)

        return x

    def rotate_query_key(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rotate query and key tensors.

        Args:
            query: Query tensor. (B, num_heads, length, size)
            key: Key tensor. (B, num_heads, length, size)
            cos: Cosine functions. (1, 1, length, size)
            sin: Sine functions. (1, 1, length, size)

        Returns:
            : Rotated query tensor. (B, num_heads, length, size)
            : Rotated key tensor. (B, num_heads, length, size)

        """
        return (
            (query * cos) + (self.rotate_half(query) * sin),
            (key * cos) + (self.rotate_half(key) * sin),
        )

    def set_cos_sin_cache(self, length: int, device: torch.device) -> None:
        """Set cos and sin cache given new sequence length.

        Args:
            length: Sequence length.
            device: Device to pin the parameters on.

        """
        self.max_len = length

        t = torch.arange(self.max_len, device=device)

        freqs = torch.einsum("i, j -> ij", t, self.inv_freq)
        embed = torch.cat((freqs, freqs), dim=-1).to(device)

        self.cached_cos = embed.cos()[None, None, :, :]
        self.cached_sin = embed.sin()[None, None, :, :]

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Apply rotary positional embeddings.

        Args:
            query: Query tensor. (B, num_heads, length, size)
            key: Key tensor. (B, num_heads, length, size)

        Returns:
            query: Rotated query tensor. (B, num_heads, length, size)
            key: Rotated key tensor. (B, num_heads, length, size)

        """
        length = query.size(1)

        if self.max_len is None or length != self.max_len:
            self.set_cos_sin_cache(length, query.device)

        query, key = self.rotate_query_key(
            query, key, self.cached_cos[:length, ...], self.cached_sin[:length, ...]
        )

        return query, key
