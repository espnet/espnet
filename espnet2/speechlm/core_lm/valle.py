#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Implementation of Vall-E: https://arxiv.org/abs/2301.02111

from typing import Tuple, Dict
import torch
import random
import logging

from espnet2.speechlm.core_lm.abs_core_lm import AbsCoreLM
from espnet2.speechlm.module.transformer import (
    TransformerDecoder,
    TransformerDecoderLevelAware,
)
from espnet2.speechlm.net_utils import (
    ce_loss,
    install_kv_cache_hook,
    logits_to_tokens,
)


class ValleLM(AbsCoreLM):
    def __init__(
        self,
        vocab_size: int,
        nq: int,
        share_emb: bool = True,
        att_unit: int = 256,
        head: int = 2,
        ar_layer: int = 4,
        nar_layer: int = 4,
        n_ctx: int = 3000,
    ):
        super(ValleLM, self).__init__()

        self.emb = torch.nn.Embedding(vocab_size, att_unit)
        self.lm_head = torch.nn.Linear(att_unit, vocab_size, bias=False)
        if share_emb:
            self.lm_head.weight = self.emb.weight

        self.ar_decoder = TransformerDecoder(
            n_ctx=n_ctx,
            n_state=att_unit,
            n_head=head,
            n_layer=ar_layer,
            causal=True
        )

        self.nar_decoder = TransformerDecoderLevelAware(
            n_ctx=n_ctx,
            n_state=att_unit,
            n_head=head,
            n_layer=nar_layer,
            n_level=nq - 1,
            causal=False
        )

        self.nq = nq

    def forward(
        self,
        dec_seq: torch.Tensor,
        dec_seq_lengths: torch.Tensor = None,
        enc_seq: torch.Tensor = None,
        enc_seq_lengths: torch.Tensor = None,
        prefix_len: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:

        assert dec_seq.dim() == 3

        # Auto-Regressive part
        input_ar = dec_seq[:, :-1, 0]
        target_ar = dec_seq[:, 1:, 0]
        h_ar = self.ar_decoder(self.emb(input_ar))
        logits_ar = self.lm_head(h_ar)

        # Non-Auto-Regressive part
        level_idx = random.randint(5, 5)
        level_idx_th = torch.ones_like(dec_seq[:, 0, 0]) * level_idx
        input_nar = dec_seq[:, 1:, :level_idx]
        target_nar = dec_seq[:, 1:, level_idx]
        h_nar = self.nar_decoder(
            self.emb(input_nar).sum(2),
            level_idx_th - 1,
        ) # [B, T, D]
        logits_nar = self.lm_head(h_nar)

        # merge and compute loss
        logits = torch.stack([logits_ar, logits_nar], dim=2)
        target = torch.stack([target_ar, target_nar], dim=2)

        loss, stats, weight = ce_loss(
            logits, 
            target,
            dec_seq_lengths - 1,
            prefix_len - 1,
        )

        stats["acc_ar"] = stats["acc_layer0"]
        stats["acc_nar"] = stats["acc_layer1"]
        stats.pop("acc_layer0")
        stats.pop("acc_layer1")

        return loss, stats, weight

    @torch.no_grad()
    def inference(
        self,
        prefix: torch.Tensor,
        opts: dict = None,
        enc_seq: torch.Tensor = None,
        suffix: torch.Tensor = None,
    ):
        # (1) initialization
        cache, hooks = install_kv_cache_hook(self.ar_decoders, {})

        # (2) auto-regressive prefix forward on first code layer
        prefix = prefix.expand(opts.nbest, -1, -1)
        suffix = suffix.expand(opts.nbest, -1, -1)
        _ = self.encode_ar(prefix[:, :-1, 0], cache=cache)  # exclude modality start

        # (3) auto-regressive loop on first code layer
        # (3.1) prepare
        minlen = int(prefix.size(1) * opts.minlenratio) if opts.minlenratio > 0 else 0
        maxlen = int(prefix.size(1) * opts.maxlenratio)
        if opts.search_algo == "teacher_force":
            assert suffix is not None
            minlen = suffix.size(1)
            maxlen = suffix.size(1)
        logging.info(f"maxlen={maxlen}, minlen={minlen}, reflen={suffix.size(1)}")

        generated = {"token": [], "score": []}
        finish_idx = torch.Tensor([-1]).expand(opts.nbest).long().to(opts.device)
        prev_tok = torch.Tensor([opts.start]).tile(opts.nbest, 1).long().to(opts.device)
        for step in range(maxlen):
            #  (3.2) search loop
            logits = self.encode_ar(prev_tok, cache=cache)
            gen_tok, gen_score = logits_to_tokens(
                logits.unsqueeze(2),
                opts,
                allow_eos=step >= minlen,
                nq_level=0,
            )
            gen_tok, gen_score = gen_tok.squeeze(2), gen_tok.squeeze(2)

            generated["token"].append(gen_tok)
            generated["score"].append(gen_score)

            if opts.search_algo == "teacher_force":
                prev_tok = suffix[:, step: step + 1, 0]
            else:
                prev_tok = gen_tok

            # (3.3) detect ended hypotheses.
            finish_idx = torch.where(
                torch.logical_and(prev_tok[:, 0] == opts.eos, finish_idx == -1),
                step,
                finish_idx,
            )

            if torch.all(torch.ge(finish_idx, 0)):
                break

        logging.info(f"Terminate at steps: {finish_idx.cpu().tolist()}")
        
        # (3.4) finalize auto-regressive
        valid_idx = finish_idx.ne(-1).nonzero(as_tuple=True)[0]
        if len(valid_idx) < prefix.size(0):
            logging.info(f"Only {len(valid_idx)} of {prefix.size(0)} are valid")
        elif len(valid_idx) == 0:
            logging.warning(f"No valid examples. Return None")
            return None, None
        
        finish_idx = finish_idx[valid_idx]
        prefix, suffix = prefix[valid_idx], suffix[valid_idx]
        gen_tokens_ar = torch.cat(generated["token"], dim=1)[valid_idx].unsqueeze(2)
        gen_scores_ar = torch.cat(generated["score"], dim=1)[valid_idx].unsqueeze(2)
        gen_tokens_ar = gen_tokens_ar[:, :finish_idx.max() + 1]
        gen_scores_ar = gen_scores_ar[:, :finish_idx.max() + 1]

        for hook in hooks:
            hook.remove()
        cache = {}

        # (4) non-auto-regressive loop on remained code layers
        # (4.1) prepare
        if opts.search_algo == "teacher_force":
            prev_tok = torch.cat([prefix[:, 1:, 0], suffix[:, :, 0]], dim=1)
        else:
            prev_tok = torch.cat([prefix[:, 1:, 0], gen_tokens_ar[:, :, 0]], dim=1)
        prev_tok = prev_tok.unsqueeze(2)

        level_idx_th = torch.arange(1, opts.nq).long().to(opts.device)
        level_idx_th = level_idx_th.unsqueeze(1).expand(-1, len(valid_idx))

        prefix_len = prefix.size(1) - 1  # <sos> excluded
        length = prefix_len + finish_idx + 1 # <eos> included

        generated = {"token": [], "score": []}
        # (4.2) search loop
        for step in range(1, opts.nq):
            logits = self.encode_nar(prev_tok, level_idx_th[step - 1], length)
            gen_tok, gen_score = logits_to_tokens(
                logits.unsqueeze(2),
                opts,
                allow_eos=False,
                nq_level=step,
            )
            gen_tok, gen_score = gen_tok.squeeze(2), gen_score.squeeze(2)

            generated["token"].append(gen_tok[:, prefix_len:])
            generated["score"].append(gen_score[:, prefix_len:])

            if opts.search_algo == "teacher_force":
                prev_tok_layer = torch.cat(
                    [prefix[:, 1:, step], suffix[:, :, step]], dim=1
                )
            else:
                prev_tok_layer = torch.cat(
                    [prefix[:, 1:, step], gen_tok[:, prefix_len:]], dim=1
                )
            prev_tok = torch.cat([prev_tok, prev_tok_layer.unsqueeze(2)], dim=2)

        # (5) compose AR and NAR results
        gen_tokens_nar = torch.stack(generated["token"], dim=2)
        gen_scores_nar = torch.stack(generated["score"], dim=2)

        gen_tokens = torch.cat([gen_tokens_ar, gen_tokens_nar], dim=2)
        gen_scores = torch.cat([gen_scores_ar, gen_scores_nar], dim=2)

        gen_tokens_list, gen_scores_list = [], []
        for b in range(len(valid_idx)):
            gen_tokens_list.append(gen_tokens[b][: finish_idx[b]])
            gen_scores_list.append(gen_scores[b][: finish_idx[b]])

        return gen_tokens_list, gen_scores_list
