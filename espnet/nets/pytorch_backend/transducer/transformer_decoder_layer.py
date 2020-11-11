"""Decoder layer definition for transformer-transducer models."""

import torch
from torch import nn

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class DecoderLayer(nn.Module):
    """Single decoder layer module for transformer-transducer models.

    Args:
        size (int): input dim
        self_attn (MultiHeadedAttention): self attention module
        feed_forward (PositionwiseFeedForward): feed forward layer module
        dropout_rate (float): dropout rate
        normalize_before (bool): whether to use layer_norm before the first block

    """

    def __init__(self, size, self_attn, feed_forward, dropout_rate):
        """Construct an DecoderLayer object."""
        super().__init__()

        self.self_attn = self_attn
        self.feed_forward = feed_forward

        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)

        self.dropout = nn.Dropout(dropout_rate)

        self.size = size

    def forward(self, tgt, tgt_mask, cache=None):
        """Compute decoded features.

        Args:
            tgt (torch.Tensor): decoded previous target features (B, Lmax, idim)
            tgt_mask (torch.Tensor): mask for tgt (B, Lmax)
            cache (torch.Tensor): cached output (B, Lmax-1, idim)

        Returns:
            tgt (torch.Tensor): decoder target features (B, Lmax, odim)
            tgt_mask (torch.Tensor): mask for tgt (B, Lmax)
        """
        residual = tgt
        tgt = self.norm1(tgt)

        if cache is None:
            tgt_q = tgt
        else:
            assert cache.shape == (
                tgt.shape[0],
                tgt.shape[1] - 1,
                self.size,
            ), f"{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"

            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]

            if tgt_mask is not None:
                tgt_mask = tgt_mask[:, -1:, :]

        tgt = residual + self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_mask))

        residual = tgt
        tgt = self.norm2(tgt)

        tgt = residual + self.dropout(self.feed_forward(tgt))

        if cache is not None:
            tgt = torch.cat([cache, tgt], dim=1)

        return tgt, tgt_mask
