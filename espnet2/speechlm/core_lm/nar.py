#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from typing import Tuple, Dict
import torch

from espnet2.speechlm.core_lm.abs_core_lm import AbsCoreLM
from espnet2.speechlm.module.module import (
    TransformerLayer,
    PositionalEncoding,
)
from espnet2.speechlm.net_utils import length_mask

class NARCoreLM(AbsCoreLM):
    def __init__(
        self,
        encoder_decoder_format: bool,
        pos_enc: str = None,
        att_unit: int = 256,
        head: int = 2,
        unit: int = 1024,
        decoder_layer: int = 4,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        flashattention: bool = False,
    ):
        super(NARCoreLM, self).__init__()

        if pos_enc == "sinusoidal":
            pos_enc_class = PositionalEncoding
        else:
            raise ValueError(f"unknown pos-enc option: {pos_enc}")
        
        self.decoders = torch.nn.ModuleList(
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
                for _ in range(decoder_layer)
            ]
        )
        self.decoder_post_ln = torch.nn.LayerNorm(att_unit)
        self.dec_pos_enc = pos_enc_class(att_unit, positional_dropout_rate)

        if encoder_decoder_format:
            raise ValueError("AR-NAR model cannot be encoder-decoder format")
        
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

        if decoder_input.dim() == 4:
            decoder_input = decoder_input.sum(dim=2)
        decoder_input = self.dec_pos_enc(decoder_input)
        for layer in self.decoders:
            decoder_input = layer(
                decoder_input,
                length_mask(decoder_input_lengths).unsqueeze(1),
                None,
                None,
                cache=cache,
            )
        decoder_output = self.decoder_post_ln(decoder_input)

        return decoder_output, decoder_input_lengths, {}