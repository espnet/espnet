# encoding: utf-8
"""Class Declaration of Transformer's Decoder."""

import chainer

import chainer.functions as F
import chainer.links as L

from espnet.nets.chainer_backend.transformer.decoder_layer import DecoderLayer
from espnet.nets.chainer_backend.transformer.embedding import PositionalEncoding
from espnet.nets.chainer_backend.transformer.layer_norm import LayerNorm
from espnet.nets.chainer_backend.transformer.mask import make_history_mask

import numpy as np


class Decoder(chainer.Chain):
    """Decoder layer.

    Args:
        odim (int): The output dimension.
        n_layers (int): Number of ecoder layers.
        n_units (int): Number of attention units.
        d_units (int): Dimension of input vector of decoder.
        h (int): Number of attention heads.
        dropout (float): Dropout rate.
        initialW (Initializer): Initializer to initialize the weight.
        initial_bias (Initializer): Initializer to initialize teh bias.

    """

    def __init__(self, odim, args, initialW=None, initial_bias=None):
        """Initialize Decoder."""
        super(Decoder, self).__init__()
        self.sos = odim - 1
        self.eos = odim - 1
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

    def make_attention_mask(self, source_block, target_block):
        """Prepare the attention mask.

        Args:
            source_block (ndarray): Source block with dimensions: (B x S).
            target_block (ndarray): Target block with dimensions: (B x T).
        Returns:
            ndarray: Mask with dimensions (B, S, T).

        """
        mask = (target_block[:, None, :] >= 0) * \
            (source_block[:, :, None] >= 0)
        # (batch, source_length, target_length)
        return mask

    def forward(self, ys_pad, source, x_mask):
        """Forward decoder.

        :param xp.array e: input token ids, int64 (batch, maxlen_out)
        :param xp.array yy_mask: input token mask, uint8  (batch, maxlen_out)
        :param xp.array source: encoded memory, float32  (batch, maxlen_in, feat)
        :param xp.array xy_mask: encoded memory mask, uint8  (batch, maxlen_in)
        :return e: decoded token score before softmax (batch, maxlen_out, token)
        :rtype: chainer.Variable
        """
        xp = self.xp
        sos = np.array([self.sos], np.int32)
        ys = [np.concatenate([sos, y], axis=0) for y in ys_pad]
        e = F.pad_sequence(ys, padding=self.eos).data
        e = xp.array(e)
        # mask preparation
        xy_mask = self.make_attention_mask(e, xp.array(x_mask))
        yy_mask = self.make_attention_mask(e, e)
        yy_mask *= make_history_mask(xp, e)

        e = self.pe(self.embed(e))
        batch, length, dims = e.shape
        e = e.reshape(-1, dims)
        source = source.reshape(-1, dims)
        for i in range(self.n_layers):
            e = self['decoders.' + str(i)](e, source, xy_mask, yy_mask, batch)
        return self.output_layer(self.output_norm(e)).reshape(batch, length, -1)

    def recognize(self, e, yy_mask, source):
        """Process recognition function."""
        e = self.forward(e, source, yy_mask)
        return F.log_softmax(e, axis=-1)
