#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Implementation of UniAudio architecture: https://arxiv.org/abs/2310.00704

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


class MultiScaleLM(AbsCoreLM):
    def __init__(
        self,
        vocab_size: int,
        nq: int,
        share_emb: bool = True,
        pos_enc: str = None,
        g_att_unit: int = 256,
        g_head: int = 2,
        g_unit: int = 1024,
        g_layer: int = 4,
        l_att_unit: int = 256,
        l_head: int = 2,
        l_unit: int = 1024,
        l_layer: int = 4,
        dropout_rate: float = 0.0,
        positional_dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        first_layer_weight: int = 1.0,
    ):
        super(MultiScaleLM, self).__init__()

        if pos_enc == "sinusoidal":
            pos_enc_class = PositionalEncoding
        else:
            raise ValueError(f"unknown pos-enc option: {pos_enc}")
        
        self.emb = torch.nn.Embedding(vocab_size, g_att_unit)
        self.lm_head = torch.nn.Linear(g_att_unit, vocab_size, bias=False)
        if share_emb:
            self.lm_head.weight = self.emb.weight

        # Global part
        self.g_decoders = torch.nn.ModuleList(
            [
                TransformerLayer(
                    att_unit=g_att_unit,
                    head=g_head,
                    unit=g_unit,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                    causal=True,
                    cross_attention=False,
                )
                for _ in range(g_layer)
            ]
        )
        self.g_post_ln = torch.nn.LayerNorm(g_att_unit)
        self.g_pos_enc = pos_enc_class(g_att_unit, positional_dropout_rate)

        # Local part
        self.l_decoders = torch.nn.ModuleList(
            [
                TransformerLayer(
                    att_unit=l_att_unit,
                    head=l_head,
                    unit=l_unit,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                    causal=True,
                    cross_attention=False,
                )
                for _ in range(l_layer)
            ]
        )
        self.l_post_ln = torch.nn.LayerNorm(l_att_unit)
        self.l_pos_enc = pos_enc_class(l_att_unit, positional_dropout_rate)
        self.placeholder = torch.nn.parameter.Parameter(
            torch.randn(1, 1, 1, l_att_unit, requires_grad=True)
        )

        # later shouls allow the local dimension is smaller than the global dimension.
        if g_att_unit != l_att_unit:
            raise ValueError("currently attention size for global and local size should be the same")

        self.nq = nq
        self.first_layer_weight = first_layer_weight

    def forward(
        self,
        decoder_input: torch.Tensor,
        decoder_input_lengths: torch.Tensor = None,
        encoder_input: torch.Tensor = None,
        encoder_input_lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        
        assert decoder_input.dim() == 3
        
        # embedding
        x = decoder_input[:, :-1]
        x = self.emb(x).sum(dim=2)

        # global
        x = self.g_pos_enc(x)
        for layer in self.g_decoders:
            x = layer(x)
        x = self.g_post_ln(x)

        # local
        B, T, _ = x.size()
        placeholder = self.placeholder.expand(B, T, -1, -1) # [B, T, 1, D]

        target = decoder_input[:, 1:]
        target_shift = torch.cat([placeholder, self.emb(target)], dim=2)[:, :, :-1] # [B, T, 1, D] - [B, T, nq, D]
        x = x.unsqueeze(2) + target_shift 
        x = x.flatten(0, 1)

        x = self.l_pos_enc(x)
        for layer in self.l_decoders:
            x = layer(x)
        x = self.l_post_ln(x)
        x = x.view(target_shift.size())

        logits = self.lm_head(x)

        loss, stats, weight = ce_loss(
            logits, target, decoder_input_lengths - 1,
            first_layer_weight=self.first_layer_weight
        )
        return loss, stats, weight

    def inference(
        self,
        prefix: torch.Tensor,
        opts: dict = None,
        suffix: torch.Tensor = None,
    ):
        raise NotImplementedError