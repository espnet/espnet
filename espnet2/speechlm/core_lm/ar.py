#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Implementation of naive langauge model with codec interleave

from typing import Tuple, Dict
import torch
import logging

from espnet2.speechlm.core_lm.abs_core_lm import AbsCoreLM, SpeechLMInferenceOptions
from espnet2.speechlm.module.module import (
    TransformerLayer,
    PositionalEncoding,
)
from espnet2.speechlm.net_utils import (
    ce_loss,
    install_kv_cache_hook,
    logits_to_tokens,
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
        dec_seq: torch.Tensor,
        dec_seq_lengths: torch.Tensor = None,
        enc_seq: torch.Tensor = None,
        enc_seq_lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:

        assert dec_seq.dim() == 3

        x = dec_seq[:, :-1]
        logits = self.encode(x)

        target = dec_seq[:, 1:]
        loss, stats, weight = ce_loss(logits, target, dec_seq_lengths - 1)

        return loss, stats, weight

    def encode(self, x: torch.Tensor, cache={}):
        x = self.emb(x).sum(dim=2)

        x = self.pos_enc(x, cache)
        for layer in self.decoders:
            x = layer(x, cache=cache)
        x = self.post_ln(x)

        logits = self.lm_head(x)
        B, T, Vnq = logits.size()
        logits = logits.view(B, T, self.nq, Vnq // self.nq)

        return logits

    @torch.no_grad()
    def inference(
        self,
        prefix: torch.Tensor,
        opts: SpeechLMInferenceOptions,
        enc_seq: torch.Tensor = None,
        suffix: torch.Tensor = None,
    ):
        # (1) initialization
        cache, hooks = install_kv_cache_hook(self.decoders, {})

        # (2) prefix forward
        prefix = prefix.expand(opts.nbest, -1, -1)
        suffix = suffix.expand(opts.nbest, -1, -1)
        _ = self.encode(prefix[:, :-1], cache=cache)  # exclude modality start

        # (3) inference loop
        minlen = int(prefix.size(1) * opts.minlenratio) if opts.minlenratio > 0 else 0
        maxlen = int(prefix.size(1) * opts.maxlenratio)
        if opts.search_algo == "teacher_force" and suffix is not None:
            minlen = suffix.size(1)
            maxlen = suffix.size(1)

        generated = {"token": [], "score": []}
        finish_idx = torch.Tensor([-1]).expand(opts.nbest).long().to(opts.device)
        prev_tok = (
            torch.Tensor([opts.start])
            .tile(opts.nbest, 1, opts.nq)
            .long()
            .to(opts.device)
        )
        for step in range(maxlen):
            #  (3.1) Search
            logits = self.encode(prev_tok, cache=cache)
            gen_tok, gen_score = logits_to_tokens(
                logits, opts, allow_eos=step >= minlen
            )

            generated["token"].append(gen_tok)
            generated["score"].append(gen_score)

            if opts.search_algo == "teacher_force":
                prev_tok = suffix[:, step].unsqueeze(1)
            else:
                prev_tok = gen_tok

            # (3.2) detect ended hypotheses.
            # TODO: fix the bug here. dimension mismatch. should be 1-d
            finish_idx = torch.where(
                torch.any(prev_tok == opts.eos),
                step,
                finish_idx,
            )

            if torch.all(torch.ge(finish_idx, 0)):
                logging.info(
                    f"Finish generation with sample lengths: {finish_idx.cpu().tolist()}"
                )
                break

        # (4) finalize
        for hook in hooks:
            hook.remove()

        generated = {
            "token": torch.cat(generated["token"], dim=1),
            "score": torch.cat(generated["score"], dim=1),
        }
        gen_tokens, gen_scores = [], []
        for b in range(opts.nbest):
            gen_tokens.append(generated["token"][b][: finish_idx[b]])
            gen_scores.append(generated["score"][b][: finish_idx[b]])

        return gen_tokens, gen_scores
