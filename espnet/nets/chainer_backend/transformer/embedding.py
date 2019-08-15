# encoding: utf-8
"""Class Declaration of Transformer's Positional Encoding."""

import chainer
import chainer.functions as F

import numpy as np


class PositionalEncoding(chainer.Chain):
    """Positional encoding module.

    :param int n_units: embedding dim
    :param float dropout: dropout rate
    :param int length: maximum input length

    """

    def __init__(self, n_units, dropout=0.1, length=5000):
        """Initialize Positional Encoding."""
        # Implementation described in the paper
        super(PositionalEncoding, self).__init__()
        self.dropout = dropout
        posi_block = np.arange(
            0, length, dtype=np.float32)[:, None]
        unit_block = np.exp(
            np.arange(0, n_units, 2, dtype=np.float32) * -(np.log(10000.) / n_units))
        self.pe = np.zeros((length, n_units), dtype=np.float32)
        self.pe[:, ::2] = np.sin(posi_block * unit_block)
        self.pe[:, 1::2] = np.cos(posi_block * unit_block)
        self.scale = np.sqrt(n_units)

    def forward(self, e):
        length = e.shape[1]
        e = e * self.scale + self.xp.array(self.pe[:length])
        return F.dropout(e, self.dropout)
