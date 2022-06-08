#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder Mix definition."""

import torch

from espnet.nets.pytorch_backend.transducer.vgg2l import VGG2L
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling


class EncoderMix(Encoder, torch.nn.Module):
    """Transformer encoder module.

    :param int idim: input dim
    :param int attention_dim: dimension of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward
    :param int num_blocks: the number of decoder blocks
    :param float dropout_rate: dropout rate
    :param float attention_dropout_rate: dropout rate in attention
    :param float positional_dropout_rate: dropout rate after adding positional encoding
    :param str or torch.nn.Module input_layer: input layer type
    :param class pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied.
        i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    :param str positionwise_layer_type: linear of conv1d
    :param int positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
    :param int padding_idx: padding_idx for input_layer=embed
    """

    def __init__(
        self,
        idim,
        attention_dim=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks_sd=4,
        num_blocks_rec=8,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        input_layer="conv2d",
        pos_enc_class=PositionalEncoding,
        normalize_before=True,
        concat_after=False,
        positionwise_layer_type="linear",
        positionwise_conv_kernel_size=1,
        padding_idx=-1,
        num_spkrs=2,
    ):
        """Construct an Encoder object."""
        super(EncoderMix, self).__init__(
            idim=idim,
            selfattention_layer_type="selfattn",
            attention_dim=attention_dim,
            attention_heads=attention_heads,
            linear_units=linear_units,
            num_blocks=num_blocks_rec,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            input_layer=input_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
            concat_after=concat_after,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            padding_idx=padding_idx,
        )
        positionwise_layer, positionwise_layer_args = self.get_positionwise_layer(
            positionwise_layer_type,
            attention_dim,
            linear_units,
            dropout_rate,
            positionwise_conv_kernel_size,
        )
        self.num_spkrs = num_spkrs
        self.encoders_sd = torch.nn.ModuleList(
            [
                repeat(
                    num_blocks_sd,
                    lambda lnum: EncoderLayer(
                        attention_dim,
                        MultiHeadedAttention(
                            attention_heads, attention_dim, attention_dropout_rate
                        ),
                        positionwise_layer(*positionwise_layer_args),
                        dropout_rate,
                        normalize_before,
                        concat_after,
                    ),
                )
                for i in range(num_spkrs)
            ]
        )

    def forward(self, xs, masks):
        """Encode input sequence.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        if isinstance(self.embed, (Conv2dSubsampling, VGG2L)):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)
        xs_sd, masks_sd = [None] * self.num_spkrs, [None] * self.num_spkrs

        for ns in range(self.num_spkrs):
            xs_sd[ns], masks_sd[ns] = self.encoders_sd[ns](xs, masks)
            xs_sd[ns], masks_sd[ns] = self.encoders(xs_sd[ns], masks_sd[ns])  # Enc_rec
            if self.normalize_before:
                xs_sd[ns] = self.after_norm(xs_sd[ns])
        return xs_sd, masks_sd

    def forward_one_step(self, xs, masks, cache=None):
        """Encode input frame.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :param List[torch.Tensor] cache: cache tensors
        :return: position embedded tensor, mask and new cache
        :rtype Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        if isinstance(self.embed, Conv2dSubsampling):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)

        new_cache_sd = []
        for ns in range(self.num_spkrs):
            if cache is None:
                cache = [
                    None for _ in range(len(self.encoders_sd) + len(self.encoders_rec))
                ]
            new_cache = []
            for c, e in zip(cache[: len(self.encoders_sd)], self.encoders_sd[ns]):
                xs, masks = e(xs, masks, cache=c)
                new_cache.append(xs)
            for c, e in zip(cache[: len(self.encoders_sd) :], self.encoders_rec):
                xs, masks = e(xs, masks, cache=c)
                new_cache.append(xs)
            new_cache_sd.append(new_cache)
            if self.normalize_before:
                xs = self.after_norm(xs)
        return xs, masks, new_cache_sd
