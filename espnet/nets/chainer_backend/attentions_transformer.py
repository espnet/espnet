# encoding: utf-8

import chainer

import chainer.functions as F
import chainer.links as L

import numpy as np

MIN_VALUE = float(np.finfo(np.float32).min)


class MultiHeadAttention(chainer.Chain):
    """Multi Head Attention Layer.

    Args:
        n_units (int)   : Number of input units.
        h (int)         : Number of attention heads.
        dropout (float): Dropout rate.
        initialW: Initializer to initialize the weight.
        initial_bias    : Initializer to initialize the bias.

    """

    def __init__(self, n_units, h=8, dropout=0.1,
                 initialW=None, initial_bias=None):
        super(MultiHeadAttention, self).__init__()
        assert n_units % h == 0
        stvd = 1. / np.sqrt(n_units)
        with self.init_scope():
            self.linear_q = L.Linear(n_units, n_units,
                                     initialW=initialW(scale=stvd),
                                     initial_bias=initial_bias(scale=stvd))
            self.linear_k = L.Linear(n_units, n_units,
                                     initialW=initialW(scale=stvd),
                                     initial_bias=initial_bias(scale=stvd))
            self.linear_v = L.Linear(n_units, n_units,
                                     initialW=initialW(scale=stvd),
                                     initial_bias=initial_bias(scale=stvd))
            self.linear_out = L.Linear(n_units, n_units,
                                       initialW=initialW(scale=stvd),
                                       initial_bias=initial_bias(scale=stvd))
        self.d_k = n_units // h
        self.h = h
        self.dropout = dropout
        self.attn = None

    def __call__(self, e_var, s_var=None, mask=None, batch=1):
        """Core function of the Multi-head attention layer.

        Args:
            e_var (chainer.Variable): Variable of input array.
            s_var (chainer.Variable): Variable of source array from encoder.
            mask (chainer.Variable) : Attention mask.
            batch (int): Batch size.

        Returns:
            chainer.Variable: Outout of multi-head attention layer.

        """
        xp = self.xp
        if s_var is None:
            # batch, head, time1/2, d_k)
            Q = self.linear_q(e_var).reshape(batch, -1, self.h, self.d_k)
            K = self.linear_k(e_var).reshape(batch, -1, self.h, self.d_k)
            V = self.linear_v(e_var).reshape(batch, -1, self.h, self.d_k)
        else:
            Q = self.linear_q(e_var).reshape(batch, -1, self.h, self.d_k)
            K = self.linear_k(s_var).reshape(batch, -1, self.h, self.d_k)
            V = self.linear_v(s_var).reshape(batch, -1, self.h, self.d_k)
        scores = F.matmul(F.swapaxes(Q, 1, 2), K.transpose(0, 2, 3, 1)) / np.sqrt(self.d_k)
        if mask is not None:
            mask = xp.stack([mask] * self.h, axis=1)
            scores = F.where(mask, scores, xp.full(scores.shape, MIN_VALUE, 'f'))
        self.attn = F.softmax(scores, axis=-1)
        p_attn = F.dropout(self.attn, self.dropout)
        x = F.matmul(p_attn, F.swapaxes(V, 1, 2))
        x = F.swapaxes(x, 1, 2).reshape(-1, self.h * self.d_k)
        return self.linear_out(x)
