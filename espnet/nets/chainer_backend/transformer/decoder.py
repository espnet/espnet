# encoding: utf-8

import chainer

import chainer.functions as F
import chainer.links as L

from espnet.nets.chainer_backend.transformer.attention import MultiHeadAttention
from espnet.nets.chainer_backend.transformer.nets_utils import FeedForwardLayer
from espnet.nets.chainer_backend.transformer.nets_utils import LayerNorm
from espnet.nets.chainer_backend.transformer.nets_utils import PositionalEncoding

import numpy as np


class DecoderLayer(chainer.Chain):
    def __init__(self, n_units, d_units=0, h=8, dropout=0.1,
                 initialW=None, initial_bias=None):
        super(DecoderLayer, self).__init__()
        with self.init_scope():
            self.self_attn = MultiHeadAttention(n_units, h, dropout=dropout,
                                                initialW=initialW,
                                                initial_bias=initial_bias)
            self.src_attn = MultiHeadAttention(n_units, h, dropout=dropout,
                                               initialW=initialW,
                                               initial_bias=initial_bias)
            self.feed_forward = FeedForwardLayer(n_units, d_units=d_units,
                                                 dropout=dropout,
                                                 initialW=initialW,
                                                 initial_bias=initial_bias)
            self.norm1 = LayerNorm(n_units)
            self.norm2 = LayerNorm(n_units)
            self.norm3 = LayerNorm(n_units)
        self.dropout = dropout

    def __call__(self, e, s, xy_mask, yy_mask, batch):
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


class Decoder(chainer.Chain):
    def __init__(self, odim, n_layers, n_units, d_units=0, h=8, dropout=0.1,
                 initialW=None, initial_bias=None):
        super(Decoder, self).__init__()
        with self.init_scope():
            self.output_norm = LayerNorm(n_units)
            self.pe = PositionalEncoding(n_units, dropout)
            stvd = 1. / np.sqrt(n_units)
            self.output_layer = L.Linear(n_units, odim, initialW=chainer.initializers.Uniform(scale=stvd),
                                         initial_bias=chainer.initializers.Uniform(scale=stvd))
        layer = L.EmbedID(odim, n_units, ignore_label=-1,
                          initialW=chainer.initializers.Normal(scale=1.0))
        self.add_link('embed.0', layer)
        for i in range(n_layers):
            name = 'decoders.' + str(i)
            layer = DecoderLayer(n_units, d_units=d_units,
                                 h=h, dropout=dropout,
                                 initialW=initialW,
                                 initial_bias=initial_bias)
            self.add_link(name, layer)
        self.n_layers = n_layers

    def __call__(self, e, yy_mask, source, xy_mask):
        e = self.pe(self['embed.0'](e))
        dims = e.shape
        e = e.reshape(-1, dims[2])
        for i in range(self.n_layers):
            e = self['decoders.' + str(i)](e, source, xy_mask, yy_mask, dims[0])
        return self.output_layer(self.output_norm(e))
