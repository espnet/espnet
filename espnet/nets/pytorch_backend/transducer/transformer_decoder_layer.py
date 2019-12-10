"""Decoder self-attention layer definition for transducer."""

import torch
from torch import nn

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class DecoderLayer(nn.Module):
    """Single self-attention decoder layer module.

    Args:
        size (int): input dim
        self_attn (MultiHeadedAttention): self attention module
        feed_forward (PositionwiseFeedForward): feed forward layer module
        dropout (float): dropout rate
        normalize_before (bool): whether to use layer_norm before the first block
        concat_after (bool): whether to concat attention layer's input and output

    """

    def __init__(self, size, self_attn, feed_forward, dropout_rate,
                 normalize_before=True, concat_after=False):
        """Construct an DecoderLayer object."""
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn

        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)

        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat = nn.Linear((size + size), size)

    def forward(self, tgt, tgt_mask, cache=None):
        """Compute decoded features.

        Args:
            tgt (torch.Tensor): decoded previous target features (B, Lmax, idim)
            tgt_mask (torch.Tensor): mask for tgt (batch, Lmax)
            cache (torch.Tensor): cached output (B, Lmax-1, idim)

        """
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            assert cache.shape == (tgt.shape[0], tgt.shape[1] - 1, self.size), \
                f"{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"

            tgt_q = tgt[:, -1, :]
            residual = residual[:, -1, :]

            if tgt_mask is not None:
                tgt_q_mask = tgt_mask[:, -1:, :]
            else:
                tgt_q_mask = None

        if self.concat_after:
            tgt_concat = torch.cat((tgt_q, self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)), dim=-1)
            x = residual + self.concat(tgt_concat)
        else:
            x = residual + self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask))
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)

        x = residual + self.dropout(self.feed_forward(x))

        if not self.normalize_before:
            x = self.norm2(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x, tgt_mask
