# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Decoder definition."""

import logging
from typing import Any, List, Sequence, Tuple

import torch
from typeguard import typechecked

from espnet2.asr.decoder.abs_decoder import AbsDecoder


class IdentityDecoder(AbsDecoder):
    """Base class of Transfomer decoder module.

    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
    """

    @typechecked
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

    def forward(
        self,
        hs_pad=None,
        hlens=None,
        ys_in_pad=None,
        ys_in_lens=None,
        return_hs=None,
        return_all_hs=None,
    ):

        return hs_pad, None
