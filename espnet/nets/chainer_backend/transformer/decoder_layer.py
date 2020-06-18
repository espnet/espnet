# encoding: utf-8
"""Class Declaration of Transformer's Decoder Block."""

import chainer

import chainer.functions as F

from espnet.nets.chainer_backend.transformer.attention import MultiHeadAttention
from espnet.nets.chainer_backend.transformer.layer_norm import LayerNorm
from espnet.nets.chainer_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)


class DecoderLayer(chainer.Chain):
    """Single decoder layer module.

    Args:
        n_units (int): Number of input/output dimension of a FeedForward layer.
        d_units (int): Number of units of hidden layer in a FeedForward layer.
        h (int): Number of attention heads.
        dropout (float): Dropout rate

    """

    def __init__(
        self, n_units, d_units=0, h=8, dropout=0.1, initialW=None, initial_bias=None
    ):
        """Initialize DecoderLayer."""
        super(DecoderLayer, self).__init__()
        with self.init_scope():
            self.self_attn = MultiHeadAttention(
                n_units,
                h,
                dropout=dropout,
                initialW=initialW,
                initial_bias=initial_bias,
            )
            self.src_attn = MultiHeadAttention(
                n_units,
                h,
                dropout=dropout,
                initialW=initialW,
                initial_bias=initial_bias,
            )
            self.feed_forward = PositionwiseFeedForward(
                n_units,
                d_units=d_units,
                dropout=dropout,
                initialW=initialW,
                initial_bias=initial_bias,
            )
            self.norm1 = LayerNorm(n_units)
            self.norm2 = LayerNorm(n_units)
            self.norm3 = LayerNorm(n_units)
        self.dropout = dropout

    def forward(self, e, s, xy_mask, yy_mask, batch):
        """Compute Encoder layer.

        Args:
            e (chainer.Variable): Batch of padded features. (B, Lmax)
            s (chainer.Variable): Batch of padded character. (B, Tmax)

        Returns:
            chainer.Variable: Computed variable of decoder.

        """
        n_e = self.norm1(e)
        n_e = self.self_attn(n_e, mask=yy_mask, batch=batch)
        e = e + F.dropout(n_e, self.dropout)

        n_e = self.norm2(e)
        n_e = self.src_attn(n_e, s_var=s, mask=xy_mask, batch=batch)
        e = e + F.dropout(n_e, self.dropout)

        n_e = self.norm3(e)
        n_e = self.feed_forward(n_e)
        e = e + F.dropout(n_e, self.dropout)
        return e
