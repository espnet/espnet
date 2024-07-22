#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Implementation of UniAudio architecture: https://arxiv.org/abs/2310.00704

import logging
from typing import Dict, Tuple

import torch

from espnet2.speechlm.core_lm.abs_core_lm import AbsCoreLM, SpeechLMInferenceOptions
from espnet2.speechlm.module.transformer import TransformerDecoder
from espnet2.speechlm.net_utils import ce_loss, install_kv_cache_hook, logits_to_tokens


class MultiScaleLM(AbsCoreLM):
    def __init__(
        self,
        vocab_size: int,
        nq: int,
        share_emb: bool = True,
        g_att_unit: int = 256,
        g_head: int = 2,
        g_layer: int = 4,
        l_att_unit: int = 256,
        l_head: int = 2,
        l_layer: int = 4,
        n_ctx: int = 3000,
        first_layer_weight: int = 1.0,
    ):
        """Initialize MultiScaleLM

        Args:
            vocab_size (int): Dimention of vocabulary.
            nq (int): Number of codes for each token / frame, usually for speech codec.
            share_emb (bool): If true, share the embedding and lm_head weight.
            g_att_unit (int): Dimention of global Transformer attention.
            g_head (int): Number of heads in global Transformer attention.
            g_layer (int): Number of layers in global Transformer.
            l_att_unit (int): Dimention of local Transformer attention.
            l_head (int): Number of heads in local Transformer attention.
            l_layer (int): Number of layers in local Transformer.
            n_ctx (int): maximum context length of global Transformer.
            first_layer_weight (int): a factor to scale the gradient for
                the first-layer codes.
        """
        super(MultiScaleLM, self).__init__()

        self.emb = torch.nn.Embedding(vocab_size, g_att_unit)
        self.lm_head = torch.nn.Linear(g_att_unit, vocab_size, bias=False)
        if share_emb:
            self.lm_head.weight = self.emb.weight

        # Global part
        self.g_decoders = TransformerDecoder(
            n_ctx=n_ctx,
            n_state=g_att_unit,
            n_head=g_head,
            n_layer=g_layer,
        )

        # Local part
        self.l_decoders = TransformerDecoder(
            n_ctx=nq,
            n_state=l_att_unit,
            n_head=l_head,
            n_layer=l_layer,
        )

        self.placeholder = torch.nn.parameter.Parameter(
            torch.randn(l_att_unit, requires_grad=True)
        )

        # later shouls allow the local dimension to be smaller than the global
        # dimension for efficient local modeling
        if g_att_unit != l_att_unit:
            raise ValueError(
                "currently attention size for global and local size should be the same"
            )

        self.nq = nq
        self.first_layer_weight = first_layer_weight

    def forward(
        self,
        dec_seq: torch.Tensor,
        dec_seq_lengths: torch.Tensor = None,
        enc_seq: torch.Tensor = None,
        enc_seq_lengths: torch.Tensor = None,
        prefix_len: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        """Auto-Regresive MultiScale forward for training

        Args:
            dec_seq (LongTensor): Batch of decoder sequences (B, T, nq).
            dec_seq_lengths (LongTensor): Lengths of batched decoder sequences (B,).
            enc_seq (LongTensor): Batch of encoder sequences (B, T, nq), keep
                the interface, may not be used.
            enc_seq_lengths (LongTensor): Lengths of batched encoder sequences (B,),
                keep the interface, may not be used.
            prefix_len (LongTensor): Lengths of condition part in dec_seq (B,).
        """
        assert dec_seq.dim() == 3

        # global
        x = dec_seq[:, :-1]
        x = self.emb(x).sum(dim=2)  # [B, T, nq, D] -> [B, T, D]
        x = self.g_decoders(x)

        # global-to-local
        B, T, _ = x.size()
        placeholder = self.placeholder.tile(B, T, 1, 1)
        target = dec_seq[:, 1:]
        target_shift = torch.cat([placeholder, self.emb(target)], dim=2)[
            :, :, :-1
        ]  # [B, T, nq, D]
        x = x.unsqueeze(2) + target_shift

        # local
        x = x.flatten(0, 1)  # [B * T, nq, D]
        x = self.l_decoders(x)
        x = x.view(target_shift.size())  # [B, T, nq, D]

        # loss
        logits = self.lm_head(x)  # [B, T, nq, V]
        loss, stats, weight = ce_loss(
            logits,
            target,
            dec_seq_lengths - 1,
            prefix_len - 1,
            first_layer_weight=self.first_layer_weight,
        )

        return loss, stats, weight

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

        # (1) global initialization
        g_cache, g_hooks = install_kv_cache_hook(self.g_decoders, {})

        # (2) Prefix forward
        prefix = prefix.expand(opts.nbest, -1, -1)
        suffix = suffix.expand(opts.nbest, -1, -1)
        prefix_emb = self.emb(prefix).sum(2)
        _ = self.g_decoders(prefix_emb, kv_cache=g_cache)

        # (3) global loop
        # (3.1) global initialization
        minlen = int(prefix.size(1) * opts.minlenratio) if opts.minlenratio > 0 else 0
        maxlen = int(prefix.size(1) * opts.maxlenratio)
        if opts.search_algo == "teacher_force":
            minlen = suffix.size(1)
            maxlen = suffix.size(1)
        logging.info(f"maxlen={maxlen}, minlen={minlen}, reflen={suffix.size(1)}")

        finish_idx = torch.Tensor([-1]).expand(opts.nbest).long().to(opts.device)

        g_generated = {"token": [], "score": []}
        g_prev_tok = (
            torch.Tensor([opts.start])
            .tile(opts.nbest, 1, opts.nq)
            .long()
            .to(opts.device)
        )
        g_prev_emb = self.emb(g_prev_tok).sum(2)  # [B, 1, D]
        for g_step in range(maxlen):
            g_hidden = self.g_decoders(g_prev_emb, kv_cache=g_cache)  # [B, 1, D]

            # (3.2) local initialization
            l_cache, l_hooks = install_kv_cache_hook(self.l_decoders, {})

            # (3.3) local loop
            l_generated = {"token": [], "score": []}
            l_prev_emb = self.placeholder.tile(opts.nbest, 1, 1)  # [B, 1, D]
            for l_step in range(opts.nq):
                l_hidden = l_prev_emb + g_hidden
                l_hidden = self.l_decoders(l_hidden, kv_cache=l_cache)
                logits = self.lm_head(l_hidden)

                gen_tok, gen_score = logits_to_tokens(
                    logits.unsqueeze(2),
                    opts,
                    allow_eos=(l_step == 0 and g_step >= minlen),
                    nq_level=l_step,
                )
                # [B, 1, 1] -> [B, 1]
                gen_tok, gen_score = gen_tok.squeeze(2), gen_score.squeeze(2)

                if opts.search_algo == "teacher_force":
                    l_prev_tok = suffix[:, g_step : g_step + 1, l_step]
                else:
                    l_prev_tok = gen_tok
                l_prev_emb = self.emb(l_prev_tok)

                l_generated["token"].append(gen_tok)
                l_generated["score"].append(gen_score)

            # (3.4) local finalize
            for hook in l_hooks:
                hook.remove()

            gen_tokens_local = torch.stack(l_generated["token"], dim=2)  # [B, 1, nq]
            gen_scores_local = torch.stack(l_generated["score"], dim=2)

            g_generated["token"].append(gen_tokens_local)
            g_generated["score"].append(gen_scores_local)

            if opts.search_algo == "teacher_force":
                g_prev_tok = suffix[:, g_step : g_step + 1]
            else:
                g_prev_tok = gen_tokens_local
            g_prev_emb = self.emb(g_prev_tok).sum(2)  # [B, 1, D]

            # (3.5) detect ended hypotheses
            finish_idx = torch.where(
                torch.logical_and(g_prev_tok[:, 0, 0] == opts.eos, finish_idx == -1),
                g_step,
                finish_idx,
            )

            if torch.all(torch.ge(finish_idx, 0)):
                break

            if g_step == maxlen - 1:
                logging.warning(
                    f"Some examples cannot finish in {maxlen} steps: {finish_idx}"
                    f"Consider increasing the maxlenratio"
                )

        logging.info(f"Finish with lengths: {finish_idx}")

        # (4) global finalize & build hypotheses
        for hook in g_hooks:
            hook.remove()

        valid_idx = finish_idx.ne(-1).nonzero(as_tuple=True)[0]
        g_generated = {
            "token": torch.cat(g_generated["token"], dim=1)[valid_idx],
            "score": torch.cat(g_generated["score"], dim=1)[valid_idx],
        }
        finish_idx = finish_idx[valid_idx]

        gen_tokens, gen_scores = [], []
        for b in range(len(valid_idx)):
            gen_tokens.append(g_generated["token"][b][: finish_idx[b]])
            gen_scores.append(g_generated["score"][b][: finish_idx[b]])
            assert not torch.any(gen_tokens[-1].eq(opts.eos))

        return gen_tokens, gen_scores
