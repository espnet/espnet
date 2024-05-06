#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Implementation of UniAudio architecture: https://arxiv.org/abs/2310.00704

from typing import Tuple, Dict
import torch
import logging

from espnet2.speechlm.core_lm.abs_core_lm import AbsCoreLM
from espnet2.speechlm.module.module import (
    TransformerLayer,
    PositionalEncoding,
)
from espnet2.speechlm.net_utils import (
    ce_loss,
    install_kv_cache_hook,
    logits_to_tokens,
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
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:

        assert dec_seq.dim() == 3

        # embedding
        x = dec_seq[:, :-1]

        # global
        x = self.encode_global(x)

        # global-to-local
        B, T, _ = x.size()
        placeholder = self.placeholder.expand(B, T, -1, -1)  # [B, T, 1, D]
        target = dec_seq[:, 1:]
        target_shift = torch.cat([placeholder, self.emb(target)], dim=2)[
            :, :, :-1
        ]
        x = x.unsqueeze(2) + target_shift
        
        # local
        x = x.flatten(0, 1)
        x = self.encode_local(x)
        x = x.view(target_shift.size())

        # loss
        logits = self.lm_head(x)
        loss, stats, weight = ce_loss(
            logits,
            target,
            dec_seq_lengths - 1,
            prefix_len,
            first_layer_weight=self.first_layer_weight,
        )
        return loss, stats, weight

    def encode_global(self, x: torch.Tensor, cache: dict={}):
        x = self.emb(x).sum(dim=2)
        x = self.g_pos_enc(x, cache=cache)
        for layer in self.g_decoders:
            x = layer(x, cache=cache)
        x = self.g_post_ln(x)
        return x

    def encode_local(self, x: torch.Tensor, cache: dict={}):
        x = self.l_pos_enc(x, cache=cache)
        for layer in self.l_decoders:
            x = layer(x, cache=cache)
        x = self.l_post_ln(x)
        return x

    @torch.no_grad()
    def inference(
        self,
        prefix: torch.Tensor,
        opts: dict = None,
        enc_seq: torch.Tensor = None,
        suffix: torch.Tensor = None,
    ):
        # (1) global initialization
        g_cache, g_hooks = install_kv_cache_hook(self.g_decoders, {})

        # (2) prefix forward
        prefix = prefix.expand(opts.nbest, -1, -1)
        suffix = suffix.expand(opts.nbest, -1, -1)
        _ = self.encode_global(prefix[:, :-1], cache=g_cache)

        # (3) global loop
        minlen = int(prefix.size(1) * opts.minlenratio) if opts.minlenratio > 0 else 0
        maxlen = int(prefix.size(1) * opts.maxlenratio)
        if opts.search_algo == "teacher_force" and suffix is not None:
            minlen = suffix.size(1)
            maxlen = suffix.size(1)
        logging.info(f"maxlen={maxlen}, minlen={minlen}, reflen={suffix.size(1)}")

        g_generated = {"token": [], "score": []}
        finish_idx = torch.Tensor([-1]).expand(opts.nbest).long().to(opts.device)
        g_prev_tok = (
            torch.Tensor([opts.start])
            .tile(opts.nbest, 1, opts.nq)
            .long()
            .to(opts.device)
        )
        for g_step in range(maxlen):
            # (3.1) global forward
            g_hidden = self.encode_global(g_prev_tok, cache=g_cache)

            # (3.2) local initialization
            l_cache, l_hooks = install_kv_cache_hook(self.l_decoders, {})

            # (3.3) local loop
            l_generated = {"token": [], "score": []}
            l_prev_tok = self.placeholder.squeeze(2)
            for l_step in range(opts.nq):
                l_hidden = l_prev_tok + g_hidden
                l_hidden = self.encode_local(l_hidden, cache=l_cache)
                logits = self.lm_head(l_hidden)
                gen_tok, gen_score = logits_to_tokens(
                    logits.unsqueeze(2),
                    opts,
                    allow_eos=(l_step == 0 and g_step >= minlen),
                    nq_level=l_step,
                )
                gen_tok, gen_score = gen_tok.squeeze(2), gen_score.squeeze(2)

                if opts.search_algo == "teacher_force":
                    l_prev_tok = suffix[:, g_step: g_step + 1, l_step]
                else:
                    l_prev_tok = gen_tok
                l_prev_tok = self.emb(l_prev_tok)

                l_generated['token'].append(gen_tok)
                l_generated['score'].append(gen_score)
            
            # (3.4) local finalize
            for hook in l_hooks:
                hook.remove()
            
            gen_tokens_local = torch.stack(l_generated['token'], dim=2)
            gen_scores_local = torch.stack(l_generated['score'], dim=2)

            g_generated['token'].append(gen_tokens_local)
            g_generated['score'].append(gen_scores_local)

            if opts.search_algo == "teacher_force":
                g_prev_tok = suffix[:, g_step: g_step + 1]
            else:
                g_prev_tok = gen_tokens_local
            
            # (3.5) detect ended hypotheses
            finish_idx = torch.where(
                torch.logical_and(g_prev_tok[:, 0, 0] == opts.eos, finish_idx == -1),
                g_step,
                finish_idx
            )

            if torch.all(torch.ge(finish_idx, 0)):
                break

            if g_step == maxlen - 1:
                logging.warning(
                    f"Some examples cannot finish in {maxlen} steps: {finish_idx}"
                    f"Consider increasing the maxlenratio"
                )
        
        logging.info(f"Finish with lengths: {finish_idx}")
        
        # global finalize
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