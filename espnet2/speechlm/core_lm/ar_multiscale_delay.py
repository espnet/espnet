#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
from typing import Dict, Tuple

import torch

from espnet2.speechlm.core_lm.abs_core_lm import AbsCoreLM, SpeechLMInferenceOptions
from espnet2.speechlm.module.transformer import TransformerDecoder
from espnet2.speechlm.net_utils import ce_loss, install_kv_cache_hook, logits_to_tokens


class ARMultiScaleDelayLM(AbsCoreLM):
    def __init__(
        self,
        vocab_size: int,
        nq: int,
        share_emb: bool = True,
        qk_norm: bool = False,
        dropout: float = 0.0,
        g_att_unit: int = 256,
        g_head: int = 2,
        g_layer: int = 4,
        l_att_unit: int = 256,
        l_head: int = 2,
        l_layer: int = 4,
        n_ctx: int = 3000,
        sos_eos: int = 5,
    ):
        """Initialize Auto-regressive LM with parallel interleave codec pattern.

        Args:
            vocab_size (int): Dimention of vocabulary.
            nq (int): Number of codes for each token / frame, usually for speech codec.
            share_emb (bool): If true, share the embedding and lm_head weight.
            qk_norm: (bool): If true, apply LayerNorm to q and k in atention.
            dropout: (float): dropout rate for attention layers.
            att_unit (int): Dimention of global Transformer attention.
            head (int): Number of heads in global Transformer attention.
            g_layer (int): Number of layers in global Transformer.
            k_layer (int): Number of layers in local Transformer.
            n_ctx (int): maximum context length of global Transformer.
        """
        super(ARMultiScaleDelayLM, self).__init__()

        self.emb = torch.nn.Embedding(vocab_size, g_att_unit)
        self.lm_head = torch.nn.Linear(l_att_unit, vocab_size, bias=False)
        if share_emb:
            assert (
                g_att_unit == l_att_unit
            ), "Cannot share embedding as g_att_unit != l_att_unit"
            self.lm_head.weight = self.emb.weight
        self.head_emb = torch.nn.Embedding(nq, l_att_unit)

        if g_att_unit != l_att_unit:
            self.g2l = torch.nn.Linear(g_att_unit, l_att_unit)
        else:
            self.g2l = torch.nn.Identity()

        self.g_decoders = TransformerDecoder(
            n_ctx=n_ctx,
            n_state=g_att_unit,
            n_head=g_head,
            n_layer=g_layer,
            qk_norm=qk_norm,
            dropout=dropout,
        )

        self.l_decoders = TransformerDecoder(
            n_ctx=nq,
            n_state=l_att_unit,
            n_head=l_head,
            n_layer=l_layer,
            qk_norm=qk_norm,
            dropout=dropout,
        )

        self.placeholder = torch.nn.parameter.Parameter(
            torch.randn(l_att_unit, requires_grad=True)
        )

        self.nq = nq
        self.n_ctx = n_ctx
        self.sos_eos = sos_eos

    def forward(
        self,
        dec_seq: torch.Tensor,
        dec_seq_lengths: torch.Tensor = None,
        enc_seq: torch.Tensor = None,
        enc_seq_lengths: torch.Tensor = None,
        prefix_len: torch.Tensor = None,
        compute_loss: bool = True,
    ) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        """Auto-Regresive LM forward for training

        Args:
            dec_seq (LongTensor): Batch of decoder sequences (B, T, nq).
            dec_seq_lengths (LongTensor): Lengths of batched decoder sequences (B,).
            enc_seq (LongTensor): Batch of encoder sequences (B, T, nq), keep the interface,
                may not be used.
            enc_seq_lengths (LongTensor): Lengths of batched encoder sequences (B,),
                keep the interface, may not be used.
            prefix_len (LongTensor): Lengths of condition part in dec_seq (B,).
            compute_loss (bool): whether to compute loss or just logits.
        """
        assert dec_seq.dim() == 3

        rank = torch.distributed.get_rank()

        # (1) delay interleave
        dec_seq_delay = self.delay_interleave(dec_seq)  # [B, T + nq - 1, nq]

        # (2) global
        x = dec_seq_delay[:, :-1]
        x = self.emb(x).sum(dim=2)  # [B, T + nq - 1, D]
        x = self.g_decoders(x)
        x = self.g2l(x)

        # (3) global to local; inverse delay interleave
        x = x.unsqueeze(2) + self.head_emb.weight.unsqueeze(0).unsqueeze(0)
        x = self.inverse_delay_interleave(x)  # [B, T, nq, D]

        B, T, _, _ = x.size()
        placeholder = self.placeholder.tile(B, T, 1, 1)
        target = dec_seq[:, 1:]
        target_emb = self.g2l(self.emb(target))
        target_shift = torch.cat([placeholder, target_emb], dim=2)[
            :, :, :-1
        ]  # [B, T, nq, D]
        x = x + target_shift

        # (4) local
        x = x.flatten(0, 1)  # [B * T, nq, D]
        x = self.l_decoders(x)
        x = x.view(target_shift.size())  # [B, T, nq, D]

        # (5) loss
        logits = self.lm_head(x)  # [B, T, nq, V]
        loss, stats, weight = ce_loss(
            logits,
            target,
            dec_seq_lengths - 1,
            prefix_len - 1,
            compute_loss=compute_loss,
        )

        return loss, logits, stats, weight

    def delay_interleave(self, dec_seq: torch.Tensor):
        B, T, nq = dec_seq.size()
        retval = (
            torch.ones(
                (B, T + nq - 1, nq),
                dtype=dec_seq.dtype,
                device=dec_seq.device,
            )
            * self.sos_eos
        )

        for n in range(nq):
            retval[:, n : n + T, n] = dec_seq[:, :, n]

        return retval

    def inverse_delay_interleave(self, x: torch.Tensor):
        B, T, nq, D = x.size()
        retval = torch.zeros(
            (B, T - nq + 1, nq, D),
            dtype=x.dtype,
            device=x.device,
        )

        for n in range(nq):
            retval[:, :, n] = x[:, n : n + T - nq + 1, n]

        return retval

    @torch.no_grad()
    def inference(
        self,
        prefix: torch.Tensor,
        opts: SpeechLMInferenceOptions,
        enc_seq: torch.Tensor = None,
        suffix: torch.Tensor = None,
    ):
        """Auto-Regresive MultiScale Inference.

        Args:
            prefix (LongTensor): Prefix part of dec_seq (B, T_dec, nq).
            opts (SpeechLMInferenceOptions): inference options.
            enc_seq (LongTensor): Encoder token sequence (B, T_enc, nq).
            suffix (LongTensor): suffix part of dec_seq (B, T_dec, nq),
                usually the target sequence for teacher-forcing.
        """

        raise NotImplementedError
