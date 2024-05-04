#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from typing import Tuple, Dict
import torch
import random

from espnet2.speechlm.core_lm.abs_core_lm import AbsCoreLM
from espnet2.speechlm.module.module import (
    TransformerLayer,
    PositionalEncoding,
)
from espnet2.speechlm.net_utils import length_mask


class ARNARCoreLM(AbsCoreLM):
    def __init__(
        self,
        encoder_decoder_format: bool,
        pos_enc: str = None,
        att_unit: int = 256,
        head: int = 2,
        unit: int = 1024,
        ar_layer: int = 4,
        nar_layer: int = 4,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        flashattention: bool = False,
    ):
        super(ARNARCoreLM, self).__init__()

        if pos_enc == "sinusoidal":
            pos_enc_class = PositionalEncoding
        else:
            raise ValueError(f"unknown pos-enc option: {pos_enc}")

        self.ar_decoder = torch.nn.ModuleList(
            [
                TransformerLayer(
                    att_unit=att_unit,
                    head=head,
                    unit=unit,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                    causal=True,
                    flashattention=flashattention,
                    cross_attention=False,
                )
                for _ in range(ar_layer)
            ]
        )
        self.ar_post_ln = torch.nn.LayerNorm(att_unit)
        self.ar_pos_enc = pos_enc_class(att_unit, positional_dropout_rate)

        self.nar_decoder = torch.nn.ModuleList(
            [
                TransformerLayer(
                    att_unit=att_unit,
                    head=head,
                    unit=unit,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                    causal=False,
                    flashattention=flashattention,
                    cross_attention=False,
                )
                for _ in range(nar_layer)
            ]
        )
        self.nar_post_ln = torch.nn.LayerNorm(att_unit)
        self.nar_pos_enc = pos_enc_class(att_unit, positional_dropout_rate)

        if encoder_decoder_format:
            raise ValueError("AR-NAR CoreLM cannot be encoder-decoder format")

        self._encoder_decoder_format = False
        self._model_dim = att_unit

    def forward(
        self,
        decoder_input: torch.Tensor,
        decoder_input_lengths: torch.Tensor = None,
        encoder_input: torch.Tensor = None,
        encoder_input_lengths: torch.Tensor = None,
        cache: Dict = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:

        assert encoder_input == None and encoder_input_lengths == None

        # (1) AR for the first layer
        ar_out = decoder_input[:, :, 0]
        for layer in self.ar_decoder:
            ar_out = layer(ar_out, None, None, None, cache=cache)

        # (2) NAR for first several layers
        # Note(Jinchuan): unlike the original Vall-E, the NAR part doesn't
        # encodes the layer index.
        layer_idx = random.randint(1, decoder_input.size(2) - 1)
        nar_out = decoder_input[:, :, :layer_idx].sum(dim=2)
        for layer in self.nar_decoder:
            nar_out = ar_out = layer(
                nar_out,
                length_mask(decoder_input_lengths).unsqueeze(1),
                None,
                None,
            )

        # (3) Combine, with shape [B, T, 2, D]
        out = torch.stack([ar_out, nar_out], dim=2)
        return out, decoder_input_lengths, {"layers": [0, layer_idx]}
