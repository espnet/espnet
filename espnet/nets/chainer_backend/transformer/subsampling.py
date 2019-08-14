# encoding: utf-8

import chainer

import chainer.functions as F
import chainer.links as L

from espnet.nets.chainer_backend.transformer.embedding import PositionalEncoding

import logging
import numpy as np


class Conv2dSubsampling(chainer.Chain):
    def __init__(self, channels, idim, dims, dropout=0.1,
                 initialW=None, initial_bias=None):
        super(Conv2dSubsampling, self).__init__()
        self.dropout = dropout
        with self.init_scope():
            # Standard deviation for Conv2D with 1 channel and kernel 3 x 3.
            n = 1 * 3 * 3
            stvd = 1. / np.sqrt(n)
            self.conv1 = L.Convolution2D(1, channels, 3, stride=2, pad=1,
                                         initialW=initialW(scale=stvd),
                                         initial_bias=initial_bias(scale=stvd))
            n = channels * 3 * 3
            stvd = 1. / np.sqrt(n)
            self.conv2 = L.Convolution2D(channels, channels, 3, stride=2, pad=1,
                                         initialW=initialW(scale=stvd),
                                         initial_bias=initial_bias(scale=stvd))
            stvd = 1. / np.sqrt(dims)
            self.out = L.Linear(idim, dims, initialW=initialW(scale=stvd),
                                initial_bias=initial_bias(scale=stvd))
            self.pe = PositionalEncoding(dims, dropout)

    def forward(self, xs, ilens):
        xs = self.xp.array(xs[:, None])
        xs = F.relu(self.conv1(xs))
        xs = F.relu(self.conv2(xs))
        batch, _, length, _ = xs.shape
        xs = self.out(F.swapaxes(xs, 1, 2).reshape(batch * length, -1))
        xs = self.pe(xs.reshape(batch, length, -1))
        # change ilens accordingly
        ilens = np.ceil(np.array(ilens, dtype=np.float32) / 2).astype(np.int)
        ilens = np.ceil(np.array(ilens, dtype=np.float32) / 2).astype(np.int)
        return xs, ilens


class LinearSampling(chainer.Chain):
    def __init__(self, idim, dims, dropout=0.1,
                 initialW=None, initial_bias=None):
        super(LinearSampling, self).__init__()
        stvd = 1. / np.sqrt(dims)
        self.dropout = dropout
        with self.init_scope():
            self.linear = L.Linear(idim, dims, initialW=initialW(scale=stvd),
                                   initial_bias=initial_bias(scale=stvd))
            self.pe = PositionalEncoding(dims, dropout)

    def forward(self, xs, ilens):
        logging.info(xs.shape)
        xs = self.linear(xs, n_batch_axes=2)
        logging.info(xs.shape)
        xs = self.pe(xs)
        return xs, ilens
