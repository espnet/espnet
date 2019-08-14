# encoding: utf-8

import chainer

from espnet.nets.chainer_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.chainer_backend.transformer.layer_norm import LayerNorm
from espnet.nets.chainer_backend.transformer.subsampling import Conv2dSubsampling
from espnet.nets.chainer_backend.transformer.subsampling import LinearSampling

import logging
import numpy as np


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

    def __init__(self, idim, args, initialW=None, initial_bias=None):
        super(Encoder, self).__init__()
        initialW = chainer.initializers.Uniform if initialW is None else initialW
        initial_bias = chainer.initializers.Uniform if initial_bias is None else initial_bias
        with self.init_scope():
            channels = 64  # Based in paper
            if args.transformer_input_layer == 'conv2d':
                idim = int(np.ceil(np.ceil(idim / 2) / 2)) * channels
                self.input_layer = Conv2dSubsampling(channels, idim, args.adim, dropout=args.dropout_rate,
                                                     initialW=initialW, initial_bias=initial_bias)
            elif args.transformer_input_layer == 'linear':
                self.input_layer = LinearSampling(idim, args.adim, initialW=initialW, initial_bias=initial_bias)
            else:
                raise ValueError('Incorrect type of input layer')
            self.norm = LayerNorm(args.adim)
        for i in range(args.elayers):
            name = 'encoders.' + str(i)
            layer = EncoderLayer(args.adim, d_units=args.eunits,
                                 h=args.aheads, dropout=args.dropout_rate,
                                 initialW=initialW,
                                 initial_bias=initial_bias)
            self.add_link(name, layer)
        self.n_layers = args.elayers

    def forward(self, e, ilens):
        """Computing Encoder layer.

        Args:
            e (chainer.Variable): Batch of padded charactor. (B, Tmax)
            ilens (chainer.Variable): Batch of length of each input batch. (B,)

        Returns:
            chainer.Variable: Computed variable of encoder.
            numpy.array: Mask.
            chainer.Variable: Batch of lengths of each encoder outputs.
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
        return self.norm(e).reshape(batch, length, -1), x_mask, ilens
