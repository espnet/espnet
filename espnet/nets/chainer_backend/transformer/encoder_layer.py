# encoding: utf-8
"""Class Declaration of Transformer's Encoder Block."""

import chainer

import chainer.functions as F

from espnet.nets.chainer_backend.transformer.attention import MultiHeadAttention
from espnet.nets.chainer_backend.transformer.layer_norm import LayerNorm
from espnet.nets.chainer_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)


class EncoderLayer(chainer.Chain):
    """Single encoder layer module.

    Args:
        n_units (int): Number of input/output dimension of a FeedForward layer.
        d_units (int): Number of units of hidden layer in a FeedForward layer.
        h (int): Number of attention heads.
        dropout (float): Dropout rate

    """

    def __init__(
        self, n_units, d_units=0, h=8, dropout=0.1, initialW=None, initial_bias=None
    ):
        """Initialize EncoderLayer."""
        super(EncoderLayer, self).__init__()
        with self.init_scope():
            self.self_attn = MultiHeadAttention(
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
        self.dropout = dropout
        self.n_units = n_units

    def forward(self, e, xx_mask, batch):
        """Forward Positional Encoding."""
        n_e = self.norm1(e)
        n_e = self.self_attn(n_e, mask=xx_mask, batch=batch)
        e = e + F.dropout(n_e, self.dropout)

        n_e = self.norm2(e)
        n_e = self.feed_forward(n_e)
        e = e + F.dropout(n_e, self.dropout)
        return e
