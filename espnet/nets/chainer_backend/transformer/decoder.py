# encoding: utf-8

import chainer

import chainer.functions as F
import chainer.links as L

from espnet.nets.chainer_backend.transformer.decoder_layer import DecoderLayer
from espnet.nets.chainer_backend.transformer.embedding import PositionalEncoding
from espnet.nets.chainer_backend.transformer.layer_norm import LayerNorm

import numpy as np


def get_topk(xp, x, k=5, axis=1):
    MIN_VALUE = float(np.finfo(np.float32).min)
    ids_list = []
    scores_list = []
    for i in range(k):
        ids = xp.argmax(x, axis=axis).astype('i')
        if axis == 0:
            scores = x[ids]
            x[ids] = MIN_VALUE
        else:
            scores = x[xp.arange(ids.shape[0]), ids]
            x[xp.arange(ids.shape[0]), ids] = MIN_VALUE
        ids_list.append(ids)
        scores_list.append(scores)
    return xp.stack(scores_list, axis=1), xp.stack(ids_list, axis=1)


class Decoder(chainer.Chain):
    """Decoder layer

    :param int odim: output dim
    :param argparse.Namespace args:  experiment setting
    """

    def __init__(self, odim, args, initialW=None, initial_bias=None):
        super(Decoder, self).__init__()
        initialW = chainer.initializers.Uniform if initialW is None else initialW
        initial_bias = chainer.initializers.Uniform if initial_bias is None else initial_bias
        with self.init_scope():
            self.output_norm = LayerNorm(args.adim)
            self.pe = PositionalEncoding(args.adim, args.dropout_rate)
            stvd = 1. / np.sqrt(args.adim)
            self.output_layer = L.Linear(args.adim, odim, initialW=initialW(scale=stvd),
                                         initial_bias=initial_bias(scale=stvd))
            self.embed = L.EmbedID(odim, args.adim, ignore_label=-1,
                                   initialW=chainer.initializers.Normal(scale=1.0))
        for i in range(args.dlayers):
            name = 'decoders.' + str(i)
            layer = DecoderLayer(args.adim, d_units=args.dunits,
                                 h=args.aheads, dropout=args.dropout_rate,
                                 initialW=initialW,
                                 initial_bias=initial_bias)
            self.add_link(name, layer)
        self.n_layers = args.dlayers

    def forward(self, e, yy_mask, source, xy_mask):
        """forward decoder

        :param xp.array e: input token ids, int64 (batch, maxlen_out)
        :param xp.array yy_mask: input token mask, uint8  (batch, maxlen_out)
        :param xp.array source: encoded memory, float32  (batch, maxlen_in, feat)
        :param xp.array xy_mask: encoded memory mask, uint8  (batch, maxlen_in)
        :return e: decoded token score before softmax (batch, maxlen_out, token)
        :rtype: chainer.Variable
        """
        e = self.pe(self.embed(e))
        dims = e.shape
        e = e.reshape(-1, dims[2])
        for i in range(self.n_layers):
            e = self['decoders.' + str(i)](e, source, xy_mask, yy_mask, dims[0])
        return self.output_layer(self.output_norm(e))

    def recognize(self, e, yy_mask, source):
        bs, lenght = e.shape
        e = self.forward(e, yy_mask, source, None)
        return F.log_softmax(e, axis=-1)
