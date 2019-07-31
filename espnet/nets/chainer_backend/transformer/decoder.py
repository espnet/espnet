# encoding: utf-8
"""Class Declaration of Transformer's Decoder."""

import chainer

import chainer.functions as F
import chainer.links as L

from espnet.nets.chainer_backend.transformer.decoder_layer import DecoderLayer
from espnet.nets.chainer_backend.transformer.embedding import PositionalEncoding
from espnet.nets.chainer_backend.transformer.layer_norm import LayerNorm

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
        """Definition of the decoder layer.

        Args:
            e (chainer.Variable): Input variable to the decoder from the encoder.
            yy_mask (chainer.Variable): Attention mask considering ys as the source and target block.
            source (List): Input sequences padded with `sos` and `pad_sequence` method.
            xy_mask (chainer.Variable): Attention mask considering ys and xs as the source/target block.

        Returns:
            chainer.Chain: Decoder layer.

        """
        e = self.pe(self.embed(e))
        dims = e.shape
        e = e.reshape(-1, dims[2])
        for i in range(self.n_layers):
            e = self['decoders.' + str(i)](e, source, xy_mask, yy_mask, dims[0])
        return self.output_layer(self.output_norm(e))

    def recognize(self, e, yy_mask, source):
        """Process predicted label."""
        bs, lenght = e.shape
        e = self.forward(e, yy_mask, source, None)
        return F.log_softmax(e, axis=-1)
