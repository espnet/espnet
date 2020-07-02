#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder self-attention layer definition."""

import torch

from torch import nn

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class EncoderLayer(nn.Module):
    """Encoder layer module.

    :param int size: input dim
    :param espnet.nets.pytorch_backend.conformer.attention.
        MultiHeadedAttention self_attn: self attention module
    :param espnet.nets.pytorch_backend.conformer.positionwise_feed_forward.
        PositionwiseFeedForward feed_forward:
        feed forward module
    :param espnet.nets.pytorch_backend.conformer.positionwise_feed_forward.
        PositionwiseFeedForward feed_forward:
        feed forward module
    :param espnet.nets.pytorch_backend.conformer.convolution.
        ConvolutionBlock feed_foreard:
        feed forward module
    :param float dropout_rate: dropout rate
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied.
        i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    :param bool rel_pos: whether to use relative positional encoding
    :param bool macaron_style: whether to use macaron style for PositionwiseFeedForward

    """

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        feed_forward2,
        conv_block,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        rel_pos=False,
        macaron_style=False,
    ):
        """Construct an EncoderLayer object."""
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward2 = feed_forward2
        self.norm1 = LayerNorm(size)  # for the first FNN module
        self.norm2 = LayerNorm(size)  # for MHA module
        self.norm3 = LayerNorm(size)  # for CNN module
        self.norm4 = LayerNorm(size)  # for the second FNN module
        self.norm5 = LayerNorm(size)
        self.conv_block = conv_block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        self.rel_pos = rel_pos
        self.macaron_style = macaron_style
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)

    def forward(self, x_input, mask, cache=None):
        """Compute encoded features.

        :param torch.Tensor x_input: encoded source features, w/o pos_emb
        (batch, max_time_in, size) or tuple((batch, max_time_in, size),
        (1, max_time_in, size))
        :param torch.Tensor mask: mask for x (batch, max_time_in)
        :param torch.Tensor cache: cache for x (batch, max_time_in - 1, size)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        if self.rel_pos:
            x, pos_emb = x_input[0], x_input[1]
        else:
            x = x_input

        # whether to use macaron style
        if self.macaron_style:
            residual = x
            if self.normalize_before:
                x = self.norm1(x)
            x = residual + 0.5 * self.dropout(self.feed_forward(x))
            if not self.normalize_before:
                x = self.norm1(x)

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm2(x)

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if self.rel_pos:
            x_att = self.self_attn(x_q, x, x, pos_emb, mask)
        else:
            x_att = self.self_attn(x_q, x, x, mask)

        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm2(x)

        # convolution module
        if self.conv_block is not None:
            residual = x
            if self.normalize_before:
                x = self.norm3(x)
            x = residual + self.dropout(self.conv_block(x))
            if not self.normalize_before:
                x = self.norm3(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm4(x)
        if self.macaron_style:
            x = residual + 0.5 * self.dropout(self.feed_forward(x))
        else:
            x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm4(x)

        if self.conv_block is not None:
            x = self.norm5(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        if self.rel_pos:
            return tuple([x, pos_emb]), mask
        else:
            return x, mask
