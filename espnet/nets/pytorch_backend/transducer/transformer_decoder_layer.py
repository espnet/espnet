"""Transformer decoder layer definition for custom Transducer model."""

from typing import Optional

import torch

from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)


class TransformerDecoderLayer(torch.nn.Module):
    """Transformer decoder layer module for custom Transducer model.

    Args:
        hdim: Hidden dimension.
        self_attention: Self-attention module.
        feed_forward: Feed forward module.
        dropout_rate: Dropout rate.

    """

    def __init__(
        self,
        hdim: int,
        self_attention: MultiHeadedAttention,
        feed_forward: PositionwiseFeedForward,
        dropout_rate: float,
    ):
        """Construct an DecoderLayer object."""
        super().__init__()

        self.self_attention = self_attention
        self.feed_forward = feed_forward

        self.norm1 = LayerNorm(hdim)
        self.norm2 = LayerNorm(hdim)

        self.dropout = torch.nn.Dropout(dropout_rate)

        self.hdim = hdim

    def forward(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
    ):
        """Compute previous decoder output sequences.

        Args:
            sequence: Transformer input sequences. (B, U, D_dec)
            mask: Transformer intput mask sequences. (B, U)
            cache: Cached decoder output sequences. (B, (U - 1), D_dec)

        Returns:
            sequence: Transformer output sequences. (B, U, D_dec)
            mask: Transformer output mask sequences. (B, U)

        """
        residual = sequence
        sequence = self.norm1(sequence)

        if cache is None:
            sequence_q = sequence
        else:
            batch = sequence.shape[0]
            prev_len = sequence.shape[1] - 1

            assert cache.shape == (
                batch,
                prev_len,
                self.hdim,
            ), f"{cache.shape} == {(batch, prev_len, self.hdim)}"

            sequence_q = sequence[:, -1:, :]
            residual = residual[:, -1:, :]

            if mask is not None:
                mask = mask[:, -1:, :]

        sequence = residual + self.dropout(
            self.self_attention(sequence_q, sequence, sequence, mask)
        )

        residual = sequence
        sequence = self.norm2(sequence)

        sequence = residual + self.dropout(self.feed_forward(sequence))

        if cache is not None:
            sequence = torch.cat([cache, sequence], dim=1)

        return sequence, mask
