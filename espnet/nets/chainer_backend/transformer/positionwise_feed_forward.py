# encoding: utf-8
"""Class Declaration of Transformer's Positionwise Feedforward."""

import chainer

import chainer.functions as F
import chainer.links as L

import numpy as np


class PositionwiseFeedForward(chainer.Chain):
    """Positionwise feed forward.

    Args:
        :param int idim: input dimenstion
        :param int hidden_units: number of hidden units
        :param float dropout_rate: dropout rate

    """

    def __init__(self, n_units, d_units=0, dropout=0.1, initialW=None, initial_bias=None):
        """Initialize PositionwiseFeedForward.

        Args:
            n_units (int): Input dimension.
            d_units (int, optional): Output dimension of hidden layer.
            dropout (float, optional): Dropout ratio.
            initialW (int, optional):  Initializer to initialize the weight.
            initial_bias (bool, optional): Initializer to initialize the bias.

        """
        super(PositionwiseFeedForward, self).__init__()
        n_inner_units = d_units if d_units > 0 else n_units * 4
        with self.init_scope():
            stvd = 1. / np.sqrt(n_units)
            self.w_1 = L.Linear(n_units, n_inner_units,
                                initialW=initialW(scale=stvd),
                                initial_bias=initial_bias(scale=stvd))
            stvd = 1. / np.sqrt(n_inner_units)
            self.w_2 = L.Linear(n_inner_units, n_units,
                                initialW=initialW(scale=stvd),
                                initial_bias=initial_bias(scale=stvd))
            self.act = F.relu
        self.dropout = dropout

    def __call__(self, e):
        """Initialize PositionwiseFeedForward.

        Args:
            e (chainer.Variable): Input variable.

        Return:
            chainer.Variable: Output variable.

        """
        e = F.dropout(self.act(self.w_1(e)), self.dropout)
        return self.w_2(e)
