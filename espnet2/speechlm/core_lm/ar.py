#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Implementation of naive langauge model with codec interleave

from typing import Tuple, Dict
import torch

from espnet2.speechlm.core_lm.abs_core_lm import AbsCoreLM
from espnet2.speechlm.module.module import (
    TransformerLayer,
    PositionalEncoding,
)
from espnet2.speechlm.net_utils import (
    ce_loss,
    install_kv_cache_hook,
)

class ARLM(AbsCoreLM):
    def __init__(
        self,
        vocab_size: int,
        nq: int,
        share_emb: bool = False,
        pos_enc: str = None,
        att_unit: int = 256,
        head: int = 2,
        unit: int = 1024,
        layer: int = 4,
        dropout_rate: float = 0.0,
        positional_dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
    ):
        super(ARLM, self).__init__()

        if pos_enc == "sinusoidal":
            pos_enc_class = PositionalEncoding
        else:
            raise ValueError(f"unknown pos-enc option: {pos_enc}")
        
        self.emb = torch.nn.Embedding(vocab_size, att_unit)
        self.lm_head = torch.nn.Linear(att_unit, vocab_size * nq, bias=False)
        if share_emb:
            raise ValueError("Embedding cannot be shared")

        self.decoders = torch.nn.ModuleList(
            [
                TransformerLayer(
                    att_unit=att_unit,
                    head=head,
                    unit=unit,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                    causal=True,
                    cross_attention=False,
                )
                for _ in range(layer)
            ]
        )
        self.post_ln = torch.nn.LayerNorm(att_unit)
        self.pos_enc = pos_enc_class(att_unit, positional_dropout_rate)

        self.nq = nq

    def forward(
        self,
        decoder_input: torch.Tensor,
        decoder_input_lengths: torch.Tensor = None,
        encoder_input: torch.Tensor = None,
        encoder_input_lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        
        assert decoder_input.dim() == 3
        
        x = decoder_input[:, :-1]
        x = self.emb(x).sum(dim=2)

        x = self.pos_enc(x)
        for layer in self.decoders:
            x = layer(x)
        x = self.post_ln(x)

        logits = self.lm_head(x)
        B, T, Vnq = logits.size()
        logits = logits.view(B, T, self.nq, Vnq // self.nq)

        target = decoder_input[:, 1:]
        loss, stats, weight = ce_loss(logits, target, decoder_input_lengths - 1)

        return loss, stats, weight

    def inference(
        self,
        prefix: torch.Tensor,
        opts: dict = None,
        suffix: torch.Tensor = None,
    ):
        raise NotImplementedError