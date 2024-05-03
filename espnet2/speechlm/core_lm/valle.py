#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Implementation of Valle: https://arxiv.org/abs/2301.02111

from typing import Tuple, Dict
import torch
import random

from espnet2.speechlm.core_lm.abs_core_lm import AbsCoreLM
from espnet2.speechlm.module.module import (
    TransformerLayer,
    LevelAwareTransformerLayer,
    PositionalEncoding,
)
from espnet2.speechlm.net_utils import (
    length_mask,
    ce_loss,
    install_kv_cache_hook,
)


class ValleLM(AbsCoreLM):
    def __init__(
        self,
        vocab_size: int,
        nq: int,
        share_emb: bool = True,
        pos_enc: str = None,
        att_unit: int = 256,
        head: int = 2,
        unit: int = 1024,
        ar_layer: int = 4,
        nar_layer: int = 4,
        dropout_rate: float = 0.0,
        positional_dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
    ):
        super(ValleLM, self).__init__()

        if pos_enc == "sinusoidal":
            pos_enc_class = PositionalEncoding
        else:
            raise ValueError(f"unknown pos-enc option: {pos_enc}")
        
        self.emb = torch.nn.Embedding(vocab_size, att_unit)
        self.lm_head = torch.nn.Linear(att_unit, vocab_size, bias=False)
        if share_emb:
            self.lm_head.weight = self.emb.weight

        # Auto-Regressive part
        self.ar_decoders = torch.nn.ModuleList(
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
                for _ in range(ar_layer)
            ]
        )
        self.ar_post_ln = torch.nn.LayerNorm(att_unit)
        self.ar_pos_enc = pos_enc_class(att_unit, positional_dropout_rate)

        # Non-Auto-Regressive part
        self.nar_decoders = torch.nn.ModuleList(
            [
                LevelAwareTransformerLayer(
                    att_unit=att_unit,
                    head=head,
                    unit=unit,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                    causal=False,
                    cross_attention=False,
                    n_level=nq - 1
                )
                for _ in range(nar_layer)
            ]
        )
        self.nar_post_ln = torch.nn.LayerNorm(att_unit)
        self.nar_pos_enc = pos_enc_class(att_unit, positional_dropout_rate)

        self.nq = nq

    def forward(
        self,
        decoder_input: torch.Tensor,
        decoder_input_lengths: torch.Tensor = None,
        encoder_input: torch.Tensor = None,
        encoder_input_lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        
        assert decoder_input.dim() == 3
        
        # Auto-Regressive part
        input_ar = decoder_input[:, :-1, 0]
        target_ar = decoder_input[:, 1:, 0]

        x_ar = self.emb(input_ar)
        x_ar = self.ar_pos_enc(x_ar)
        for layer in self.ar_decoders:
            x_ar = layer(x_ar)
        x_ar = self.ar_post_ln(x_ar)
        
        logits_ar = self.lm_head(x_ar)
        loss_ar, stats_ar, weight_ar = ce_loss(
            logits_ar.unsqueeze(2),
            target_ar.unsqueeze(2), 
            decoder_input_lengths -1
        )

        # Non-Auto-Regressive part
        level_idx = random.randint(1, self.nq - 1)
        input_nar = decoder_input[:, 1:, :level_idx]
        target_nar = decoder_input[:, 1:, level_idx]

        mask = length_mask(decoder_input_lengths - 1).unsqueeze(1)
        level_idx_th = torch.Tensor([level_idx]).long().to(mask.device).expand(input_nar.size(0))

        x_nar = self.emb(input_nar).sum(dim=2)
        x_nar = self.nar_pos_enc(x_nar)
        for layer in self.nar_decoders:
            x_nar = layer(x_nar, level_idx_th - 1, mask)
        x_nar = self.nar_post_ln(x_nar)

        logits_nar = self.lm_head(x_nar)
        loss_nar, stats_nar, weight_nar = ce_loss(
            logits_nar.unsqueeze(2),
            target_nar.unsqueeze(2), 
            decoder_input_lengths - 1
        )

        # Aggregate
        loss = (loss_ar + loss_nar) / 2
        weight = weight_ar + weight_nar
        stats = {
            "loss": loss.item(),
            "loss_ar": loss_ar.item(),
            "loss_nar": loss_nar.item(),
            "acc_ar": stats_ar[f"acc_layer0"],
            "acc_nar": stats_nar[f"acc_layer0"],
            "weight": weight,
        }
        return loss, stats, weight

    def inference(
        self,
        prefix: torch.Tensor,
        opts: dict = None,
        suffix: torch.Tensor = None,
    ):
        raise NotImplementedError