#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Implementation of Vall-E: https://arxiv.org/abs/2301.02111

import logging
from typing import Dict, Tuple

import torch

from espnet2.speechlm.core_lm.abs_core_lm import AbsCoreLM, SpeechLMInferenceOptions
from espnet2.speechlm.module.transformer import TransformerDecoder
from espnet2.speechlm.module.valle import ValleNARDecoder
from espnet2.speechlm.net_utils import (
    ce_loss,
    install_kv_cache_hook,
    length_mask,
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
        """Initialize Vall-E model

        Args:
            vocab_size (int): Dimention of vocabulary.
            nq (int): Number of codes for each token / frame, usually for speech codec.
            share_emb (bool): If true, share the embedding and lm_head weight.
            att_unit (int): Dimention of Transformer attention.
            head (int): Number of heads in Transformer attention.
            ar_layer (int): Number of layers in AR Transformer.
            nar_layer (int): Number of layers in NAR Transformer.
            n_ctx (int): maximum context length of AR & NAR Transformer.
        """
        super(ValleLM, self).__init__()

        self.emb = torch.nn.Embedding(vocab_size, att_unit)
        self.lm_head = torch.nn.Linear(att_unit, vocab_size, bias=False)
        if share_emb:
            self.lm_head.weight = self.emb.weight

        self.ar_decoder = TransformerDecoder(
            n_ctx=n_ctx, n_state=att_unit, n_head=head, n_layer=ar_layer, causal=True
        )

        self.nar_decoder = ValleNARDecoder(
            n_level=nq - 1,
            n_ctx=n_ctx,
            n_state=att_unit,
            n_head=head,
            n_layer=nar_layer,
            causal=False,
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
        """Vall-E forward for training

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

        batch_size = dec_seq.size(0)
        dec_seq_emb = self.emb(dec_seq)  # [B, T, nq, D]

        # Auto-Regressive part
        input_ar_emb = self.prepare_input(dec_seq_emb, prefix_len, 1)[
            :, :-1
        ]  # [B, T, D]
        target_ar = dec_seq[:, 1:, 0]
        h_ar = self.ar_decoder(input_ar_emb)
        logits_ar = self.lm_head(h_ar)  # [B, T, V]

        # Non-Auto-Regressive part
        level_idx_th = torch.randint(
            1, self.nq, (batch_size,), device=dec_seq.device
        ).long()
        input_nar_emb = self.prepare_input(dec_seq_emb, prefix_len, level_idx_th)[
            :, 1:
        ]  # [B, T, V]
        batch_idx = torch.arange(batch_size, device=dec_seq.device)
        target_nar = dec_seq[batch_idx, 1:, level_idx_th]
        h_nar = self.nar_decoder(input_nar_emb, level_idx_th - 1)
        logits_nar = self.lm_head(h_nar)  # [B, T, V]

        # merge and compute loss
        logits = torch.stack([logits_ar, logits_nar], dim=2)  # [B, T, 2, V]
        target = torch.stack([target_ar, target_nar], dim=2)  # [B, T, 2]

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

    def prepare_input(self, dec_seq_emb, prefix_len, level):
        # (1) level mask, [B, 1, nq, 1], True is to include
        if isinstance(level, int):
            level = torch.ones_like(dec_seq_emb[:, 0, 0, 0]) * level
        level_mask = length_mask(level, maxlen=self.nq).bool()
        level_mask = level_mask.unsqueeze(1).unsqueeze(3)

        # (2) prefix mask, [B, T, 1, 1], True is the prefix
        prefix_mask = length_mask(prefix_len, maxlen=dec_seq_emb.size(1)).bool()
        prefix_mask = prefix_mask.unsqueeze(2).unsqueeze(3)

        # (3) mask and then sum in nq-axis.
        mask = torch.logical_or(level_mask, prefix_mask)
        return dec_seq_emb.masked_fill(~mask, 0.0).sum(2)

    @torch.no_grad()
    def inference(
        self,
        prefix: torch.Tensor,
        opts: SpeechLMInferenceOptions,
        enc_seq: torch.Tensor = None,
        suffix: torch.Tensor = None,
    ):
        """Vall-E Inference.

        Args:
            prefix (LongTensor): Prefix part of dec_seq (B, T, nq).
            opts (SpeechLMInferenceOptions): inference options.
            enc_seq (LongTensor): Encoder token sequence (B, T, nq).
            suffix (LongTensor): suffix part of dec_seq (B, T, nq),
                usually the target sequence for teacher-forcing.
        """

        # (1) initialization
        cache, hooks = install_kv_cache_hook(self.ar_decoder, {})

        # (2) auto-regressive prefix forward on first code layer
        prefix = prefix.expand(opts.nbest, -1, -1)
        suffix = suffix.expand(opts.nbest, -1, -1)
        prefix_emb = self.emb(prefix).sum(dim=2)  # [B, T, D]
        _ = self.ar_decoder(prefix_emb, kv_cache=cache)

        # (3) auto-regressive loop on first code layer
        # (3.1) AR initialization
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
            #  (3.2) AR loop
            prev_emb = self.emb(prev_tok)  # [B, 1, D]
            h_ar = self.ar_decoder(prev_emb, kv_cache=cache)
            logits = self.lm_head(h_ar)  # [B, 1, V]
            gen_tok, gen_score = logits_to_tokens(
                logits.unsqueeze(2),
                opts,
                allow_eos=step >= minlen,
                nq_level=0,
            )
            # [B, 1, 1] -> [B, 1]
            gen_tok, gen_score = gen_tok.squeeze(2), gen_tok.squeeze(2)

            generated["token"].append(gen_tok)
            generated["score"].append(gen_score)

            if opts.search_algo == "teacher_force":
                prev_tok = suffix[:, step : step + 1, 0]
            else:
                prev_tok = gen_tok  # [B, 1]

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
            logging.warning("No valid examples. Return None")
            return None, None

        finish_idx = finish_idx[valid_idx]
        prefix_emb, suffix = prefix_emb[valid_idx], suffix[valid_idx]
        gen_tokens_ar = torch.cat(generated["token"], dim=1)[valid_idx].unsqueeze(
            2
        )  # [B, T, 1]
        gen_scores_ar = torch.cat(generated["score"], dim=1)[valid_idx].unsqueeze(2)
        gen_tokens_ar = gen_tokens_ar[:, : finish_idx.max() + 1]  # to include <sos>
        gen_scores_ar = gen_scores_ar[:, : finish_idx.max() + 1]

        for hook in hooks:
            hook.remove()
        cache = {}

        # (4) non-auto-regressive loop on the remained code layers
        # (4.1) NAR initialization
        if opts.search_algo == "teacher_force":
            prev_tok = suffix[:, :, 0]
        else:
            prev_tok = gen_tokens_ar[:, :, 0]
        start_emb = self.emb.weight[opts.start].tile(opts.nbest, 1, 1)  # [B, 1, D]
        prev_emb = torch.cat(
            [prefix_emb[:, 1:], start_emb, self.emb(prev_tok)], dim=1
        )  # [B, T, D]

        ones = torch.ones_like(valid_idx)
        generated = {"token": [], "score": []}
        # (4.2) NAR loop
        for step in range(1, opts.nq):
            h_nar = self.nar_decoder(prev_emb, ones * step - 1)  # [B, T, D]
            logits = self.lm_head(h_nar)  # [B, T, V]
            gen_tok, gen_score = logits_to_tokens(
                logits.unsqueeze(2),
                opts,
                allow_eos=False,
                nq_level=step,
            )
            gen_tok, gen_score = gen_tok.squeeze(2), gen_score.squeeze(2)  # [B, T]

            generated["token"].append(gen_tok[:, prefix.size(1) :])
            generated["score"].append(gen_score[:, prefix.size(1) :])

            if opts.search_algo == "teacher_force":
                prev_tok = suffix[:, :, step]
            else:
                prev_tok = gen_tok
            prev_emb[:, prefix.size(1) :] += self.emb(prev_tok)  # [B, T, D]
            prev_emb[:, prefix.size(1) - 1 : prefix.size(1)] += start_emb

        # (5) combine AR and NAR results
        gen_tokens_nar = torch.stack(generated["token"], dim=2)  # [B, T, nq]
        gen_scores_nar = torch.stack(generated["score"], dim=2)

        gen_tokens = torch.cat([gen_tokens_ar, gen_tokens_nar], dim=2)  # [B, T, nq]
        gen_scores = torch.cat([gen_scores_ar, gen_scores_nar], dim=2)

        gen_tokens_list, gen_scores_list = [], []
        for b in range(len(valid_idx)):
            gen_tokens_list.append(gen_tokens[b][: finish_idx[b]])
            gen_scores_list.append(gen_scores[b][: finish_idx[b]])

        return gen_tokens_list, gen_scores_list
