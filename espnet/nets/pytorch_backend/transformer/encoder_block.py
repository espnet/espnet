#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""

import torch
import logging

from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.attention import RelBlockMultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.lightconv import LightweightConvolution
from espnet.nets.pytorch_backend.transformer.lightconv2d import LightweightConvolution2D
from espnet.nets.pytorch_backend.transformer.dynamic_conv import DynamicConvolution
from espnet.nets.pytorch_backend.transformer.dynamic_conv2d import DynamicConvolution2D
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder_layer_xl import EncoderLayerXL, EncoderLayerKermitXL
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import PositionwiseFeedForward
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling


class EncoderXL(torch.nn.Module):
    """Transformer encoder module.

    :param int idim: input dim
    :param int attention_dim: dimention of attention
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
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    :param str positionwise_layer_type: linear of conv1d
    :param int positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
    :param int padding_idx: padding_idx for input_layer=embed
    """

    def __init__(self, idim, odim, 
                 selfattention_layer_type="selfattn",
                 attention_dim=256,
                 attention_heads=4,
                 conv_wshare=4,
                 conv_kernel_length=11,
                 conv_usebias=False,
                 linear_units=2048,
                 num_blocks=6,
                 dropout_rate=0.1,
                 positional_dropout_rate=0.1,
                 attention_dropout_rate=0.0,
                 input_layer_audio="conv2d",
                 input_layer_char="embed",
                 pos_enc_class=PositionalEncoding,
                 normalize_before=True,
                 concat_after=False,
                 positionwise_layer_type="linear",
                 positionwise_conv_kernel_size=1,
                 xl_block_length=32,
                 padding_idx=-1):
        """Construct an Encoder object."""
        super(EncoderXL, self).__init__()

        self.pos_enc = pos_enc_class(attention_dim, positional_dropout_rate, reverse=True)
        if input_layer_audio == "linear":
            self.embed_audio = torch.nn.Sequential(
                torch.nn.Linear(idim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        elif input_layer_audio == "conv2d":
            #self.embed_audio = Conv2dSubsampling(idim, attention_dim, dropout_rate, skip_pe=True)
            self.embed_audio = Conv2dSubsampling(idim, attention_dim, dropout_rate, skip_pe=False)
        elif input_layer_audio == "embed":
            self.embed_audio = torch.nn.Sequential(
                torch.nn.Embedding(idim, attention_dim, padding_idx=padding_idx),
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        elif isinstance(input_layer_audio, torch.nn.Module):
            self.embed_audio = torch.nn.Sequential(
                input_layer_audio,
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer_audio is None:
            self.embed_audio = None
            #self.embed_audio = torch.nn.Sequential(
            #    pos_enc_class(attention_dim, positional_dropout_rate)
            #)
        else:
            raise ValueError("unknown input_layer_audio: " + input_layer_audio)


        if input_layer_char == "embed":
            if selfattention_layer_type == "relselfattn":
                self.embed_char = torch.nn.Sequential(
                    torch.nn.Embedding(odim, attention_dim),
                    pos_enc_class(attention_dim, positional_dropout_rate)
                )
            else:
                self.embed_char = torch.nn.Sequential(
                    torch.nn.Embedding(odim, attention_dim),
                    pos_enc_class(attention_dim, positional_dropout_rate)
                )
        elif input_layer_char == "linear":
            self.embed_char = torch.nn.Sequential(
                torch.nn.Linear(odim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        elif isinstance(input_layer_char, torch.nn.Module):
            self.embed_char = torch.nn.Sequential(
                input_layer_char,
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        elif input_layer_char is None:
            self.embed_char = None
        else:
            raise NotImplementedError("only `embed` or torch.nn.Module is supported.")

        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (attention_dim, linear_units, dropout_rate)
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (attention_dim, linear_units, positionwise_conv_kernel_size, dropout_rate)
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (attention_dim, linear_units, positionwise_conv_kernel_size, dropout_rate)
        else:
            raise NotImplementedError("Support only linear or conv1d.")
        if selfattention_layer_type == "relselfattn":
            logging.warning('encoder self-attention layer type = rel-self-attention')
            if num_blocks == 0:
                self.encoders = None
            else:
                self.encoders = repeat(
                    num_blocks,
                    lambda lnum: EncoderLayerXL(
                        attention_dim,
                        RelBlockMultiHeadedAttention(attention_heads, attention_dim, attention_dropout_rate, pos_emb=self.pos_enc, block_len=xl_block_length),
                        positionwise_layer(*positionwise_layer_args),
                        dropout_rate,
                        normalize_before,
                        concat_after
                    )
                )
        elif selfattention_layer_type == "relselfattn_kermit":
            logging.warning('encoder self-attention layer type = rel-self-attention-kermit')
            self.encoders = repeat(
                num_blocks,
                lambda lnum: EncoderLayerKermitXL(
                    attention_dim,
                    RelBlockMultiHeadedAttention(attention_heads, attention_dim, attention_dropout_rate, pos_emb=self.pos_enc, block_len=xl_block_length),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before,
                    concat_after
                )
            )
        elif selfattention_layer_type == "lightconv":
            logging.warning('encoder self-attention layer type = lightweight convolution')
            self.encoders = repeat(
                num_blocks,
                lambda lnum: EncoderLayer(
                    attention_dim,
                    LightweightConvolution(conv_wshare, attention_dim, attention_dropout_rate, conv_kernel_length, lnum, use_bias=conv_usebias),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before,
                    concat_after
                )
            )
        elif selfattention_layer_type == "lightconv2d":
            logging.warning('encoder self-attention layer type = lightweight convolution 2-dimentional')
            self.encoders = repeat(
                num_blocks,
                lambda lnum: EncoderLayer(
                    attention_dim,
                    LightweightConvolution2D(conv_wshare, attention_dim, attention_dropout_rate, conv_kernel_length, lnum, use_bias=conv_usebias),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before,
                    concat_after
                )
            )
        elif selfattention_layer_type == "dynamicconv":
            logging.warning('encoder self-attention layer type = dynamic convolution')
            self.encoders = repeat(
                num_blocks,
                lambda lnum: EncoderLayer(
                    attention_dim,
                    DynamicConvolution(conv_wshare, attention_dim, attention_dropout_rate, conv_kernel_length, lnum, use_bias=conv_usebias),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before,
                    concat_after
                )
            )
        elif selfattention_layer_type == "dynamicconv2d":
            logging.warning('encoder self-attention layer type = dynamic convolution 2-dimentional')
            self.encoders = repeat(
                num_blocks,
                lambda lnum: EncoderLayer(
                    attention_dim,
                    DynamicConvolution2D(conv_wshare, attention_dim, attention_dropout_rate, conv_kernel_length, lnum, use_bias=conv_usebias),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before,
                    concat_after
                )
            )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

    def forward(self, xs, masks, km, km_mask, bl):
        """Embed positions in tensor.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        if isinstance(self.embed_audio, Conv2dSubsampling):
            xs, masks = self.embed_audio(xs, masks)
        else:
            if self.embed_audio is not None:
                xs = self.embed_audio(xs)
        
        if self.embed_char is not None:
            km = self.embed_char(km)

        if self.encoders is not None:
            xs, masks, ys, ys_mask, bl = self.encoders(xs, masks, km, km_mask, bl)
            if self.normalize_before:
                xs = self.after_norm(xs)
                if ys is not None:
                    ys = self.after_norm(ys)
            return xs, masks, ys, ys_mask, bl
        else:
            return xs, masks, None, None, bl
