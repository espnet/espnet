# encoding: utf-8
"""Class Declaration of Transformer's Encoder."""

import logging

import chainer
import numpy as np
from chainer import links as L

from espnet.nets.chainer_backend.transformer.embedding import \
    PositionalEncoding
from espnet.nets.chainer_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.chainer_backend.transformer.layer_norm import LayerNorm
from espnet.nets.chainer_backend.transformer.mask import make_history_mask
from espnet.nets.chainer_backend.transformer.subsampling import (
    Conv2dSubsampling, LinearSampling)


class Encoder(chainer.Chain):
    """Encoder.

    Args:
        input_type(str):
            Sampling type. `input_type` must be `conv2d` or 'linear' currently.
        idim (int): Dimension of inputs.
        n_layers (int): Number of encoder layers.
        n_units (int): Number of input/output dimension of a FeedForward layer.
        d_units (int): Number of units of hidden layer in a FeedForward layer.
        h (int): Number of attention heads.
        dropout (float): Dropout rate

    """

    def __init__(
        self,
        idim,
        attention_dim=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        input_layer="conv2d",
        pos_enc_class=PositionalEncoding,
        initialW=None,
        initial_bias=None,
    ):
        """Initialize Encoder.

        Args:
            idim (int): Input dimension.
            args (Namespace): Training config.
            initialW (int, optional):  Initializer to initialize the weight.
            initial_bias (bool, optional): Initializer to initialize the bias.

        """
        super(Encoder, self).__init__()
        initialW = chainer.initializers.Uniform if initialW is None else initialW
        initial_bias = (
            chainer.initializers.Uniform if initial_bias is None else initial_bias
        )
        self.do_history_mask = False
        with self.init_scope():
            self.conv_subsampling_factor = 1
            channels = 64  # Based in paper
            if input_layer == "conv2d":
                idim = int(np.ceil(np.ceil(idim / 2) / 2)) * channels
                self.input_layer = Conv2dSubsampling(
                    channels,
                    idim,
                    attention_dim,
                    dropout=dropout_rate,
                    initialW=initialW,
                    initial_bias=initial_bias,
                )
                self.conv_subsampling_factor = 4
            elif input_layer == "linear":
                self.input_layer = LinearSampling(
                    idim, attention_dim, initialW=initialW, initial_bias=initial_bias
                )
            elif input_layer == "embed":
                self.input_layer = chainer.Sequential(
                    L.EmbedID(idim, attention_dim, ignore_label=-1),
                    pos_enc_class(attention_dim, positional_dropout_rate),
                )
                self.do_history_mask = True
            else:
                raise ValueError("unknown input_layer: " + input_layer)
            self.norm = LayerNorm(attention_dim)
        for i in range(num_blocks):
            name = "encoders." + str(i)
            layer = EncoderLayer(
                attention_dim,
                d_units=linear_units,
                h=attention_heads,
                dropout=attention_dropout_rate,
                initialW=initialW,
                initial_bias=initial_bias,
            )
            self.add_link(name, layer)
        self.n_layers = num_blocks

    def forward(self, e, ilens):
        """Compute Encoder layer.

        Args:
            e (chainer.Variable): Batch of padded character. (B, Tmax)
            ilens (chainer.Variable): Batch of length of each input batch. (B,)

        Returns:
            chainer.Variable: Computed variable of encoder.
            numpy.array: Mask.
            chainer.Variable: Batch of lengths of each encoder outputs.

        """
        if isinstance(self.input_layer, Conv2dSubsampling):
            e, ilens = self.input_layer(e, ilens)
        else:
            e = self.input_layer(e)
        batch, length, dims = e.shape
        x_mask = np.ones([batch, length])
        for j in range(batch):
            x_mask[j, ilens[j] :] = -1
        xx_mask = (x_mask[:, None, :] >= 0) * (x_mask[:, :, None] >= 0)
        xx_mask = self.xp.array(xx_mask)
        if self.do_history_mask:
            history_mask = make_history_mask(self.xp, x_mask)
            xx_mask *= history_mask
        logging.debug("encoders size: " + str(e.shape))
        e = e.reshape(-1, dims)
        for i in range(self.n_layers):
            e = self["encoders." + str(i)](e, xx_mask, batch)
        return self.norm(e).reshape(batch, length, -1), x_mask, ilens
