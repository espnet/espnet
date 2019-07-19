# encoding: utf-8

import chainer

import chainer.functions as F
import chainer.links as L

from espnet.nets.chainer_backend.attentions_transformer import MultiHeadAttention
from espnet.nets.chainer_backend.nets_utils_transformer import FeedForwardLayer
from espnet.nets.chainer_backend.nets_utils_transformer import LayerNorm
from espnet.nets.chainer_backend.nets_utils_transformer import PositionalEncoding

import logging
import numpy as np


class Conv2dSubsampling(chainer.Chain):
    def __init__(self, channels, idim, dims, dropout=0.1,
                 initialW=None, initial_bias=None):
        super(Conv2dSubsampling, self).__init__()
        n = 1 * 3 * 3
        stvd = 1. / np.sqrt(n)
        layer = L.Convolution2D(1, channels, 3, stride=2, pad=1,
                                initialW=initialW(scale=stvd),
                                initial_bias=initial_bias(scale=stvd))
        self.add_link('conv.0', layer)
        n = channels * 3 * 3
        stvd = 1. / np.sqrt(n)
        layer = L.Convolution2D(channels, channels, 3, stride=2, pad=1,
                                initialW=initialW(scale=stvd),
                                initial_bias=initial_bias(scale=stvd))
        self.add_link('conv.2', layer)
        stvd = 1. / np.sqrt(dims)
        layer = L.Linear(idim, dims, initialW=initialW(scale=stvd),
                         initial_bias=initial_bias(scale=stvd))
        self.add_link('out.0', layer)
        self.dropout = dropout
        with self.init_scope():
            self.pe = PositionalEncoding(dims, dropout)

    def __call__(self, xs, ilens):
        xs = F.expand_dims(xs, axis=1).data
        xs = F.relu(self['conv.{}'.format(0)](xs))
        xs = F.relu(self['conv.{}'.format(2)](xs))
        batch, _, length, _ = xs.shape
        xs = self['out.0'](F.swapaxes(xs, 1, 2).reshape(batch * length, -1))
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

    def __call__(self, xs, ilens):
        logging.info(xs.shape)
        xs = self.linear(xs, n_batch_axes=2)
        logging.info(xs.shape)
        xs = self.pe(xs)
        return xs, ilens


class EncoderLayer(chainer.Chain):
    def __init__(self, n_units, d_units=0, h=8, dropout=0.1,
                 initialW=None, initial_bias=None):
        super(EncoderLayer, self).__init__()
        with self.init_scope():
            self.self_attn = MultiHeadAttention(n_units, h, dropout=dropout,
                                                initialW=initialW,
                                                initial_bias=initial_bias)
            self.feed_forward = FeedForwardLayer(n_units, d_units=d_units,
                                                 dropout=dropout,
                                                 initialW=initialW,
                                                 initial_bias=initial_bias)
            self.norm1 = LayerNorm(n_units)
            self.norm2 = LayerNorm(n_units)
        self.dropout = dropout
        self.n_units = n_units

    def __call__(self, e, xx_mask, batch):
        n_e = self.norm1(e)
        n_e = self.self_attn(n_e, mask=xx_mask, batch=batch)
        e = e + F.dropout(n_e, self.dropout)

        n_e = self.norm2(e)
        n_e = self.feed_forward(n_e)
        e = e + F.dropout(n_e, self.dropout)
        return e


class Encoder(chainer.Chain):
    """Encoder.

    Args:
        input_type(str): Sampling type. `input_type` must be `conv2d` or 'linear' currently.
        idim (int): Dimension of inputs.
        n_layers (int): Number of encoder layers.
        n_units (int): Number of input/output dimension of a FeedForward layer.
        d_units (int): Number of units of hidden layer in a FeedForward layer.
        h (int): Number of attention heads.
        dropout (float): Dropout rate

    """

    def __init__(self, input_type, idim, n_layers, n_units, d_units=0, h=8, dropout=0.1,
                 initialW=None, initial_bias=None):
        super(Encoder, self).__init__()
        with self.init_scope():
            channels = 64  # Based in paper
            if input_type == 'conv2d':
                idim = int(np.ceil(np.ceil(idim / 2) / 2)) * channels
                self.input_layer = Conv2dSubsampling(channels, idim, n_units, dropout=dropout,
                                                     initialW=initialW, initial_bias=initial_bias)
            elif input_type == 'linear':
                self.input_layer = LinearSampling(idim, n_units, initialW=initialW, initial_bias=initial_bias)
            else:
                raise ValueError('Incorrect type of input layer')
            self.norm = LayerNorm(n_units)
        for i in range(n_layers):
            name = 'encoders.' + str(i)
            layer = EncoderLayer(n_units, d_units=d_units,
                                 h=h, dropout=dropout,
                                 initialW=initialW,
                                 initial_bias=initial_bias)
            self.add_link(name, layer)
        self.n_layers = n_layers

    def __call__(self, e, ilens):
        """Computing Encoder layer.

        Args:
            e (chainer.Variable): Batch of padded charactor. (B, Tmax)
            ilens (chainer.Variable): Batch of length of each input batch. (B,)

        Returns:
            chainer.Variable: Computed variable of encoder.
            numpy.array: Mask.
            chainer.Variable: `ilens`.

        """
        e, ilens = self.input_layer(e, ilens)
        batch, length, dims = e.shape
        x_mask = np.ones([batch, length])
        for j in range(batch):
            x_mask[j, ilens[j]:] = -1
        xx_mask = (x_mask[:, None, :] >= 0) * (x_mask[:, :, None] >= 0)
        xx_mask = self.xp.array(xx_mask)
        logging.debug('encoders size: ' + str(e.shape))
        e = e.reshape(-1, dims)
        for i in range(self.n_layers):
            e = self['encoders.' + str(i)](e, xx_mask, batch)
        return self.norm(e), x_mask, ilens
