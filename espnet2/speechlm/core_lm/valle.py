#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Implementation of Vall-E: https://arxiv.org/abs/2301.02111

from typing import Tuple, Dict
import torch
import random
import logging

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
    logits_to_tokens,
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
        dec_seq: torch.Tensor,
        dec_seq_lengths: torch.Tensor = None,
        enc_seq: torch.Tensor = None,
        enc_seq_lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        
        assert dec_seq.dim() == 3
        
        # Auto-Regressive part
        input_ar = dec_seq[:, :-1, 0]
        target_ar = dec_seq[:, 1:, 0]
        logits_ar = self.encode_ar(input_ar)

        loss_ar, stats_ar, weight_ar = ce_loss(
            logits_ar.unsqueeze(2),
            target_ar.unsqueeze(2), 
            dec_seq_lengths - 1
        )

        # Non-Auto-Regressive part
        level_idx = random.randint(1, self.nq - 1)
        level_idx_th = torch.Tensor([level_idx]).long().to(dec_seq.device).expand(dec_seq.size(0))
        input_nar = dec_seq[:, 1:, :level_idx]
        target_nar = dec_seq[:, 1:, level_idx]

        logits_nar = self.encode_nar(
            input_nar,
            level_idx_th,
            dec_seq_lengths - 1,
        )
        
        loss_nar, stats_nar, weight_nar = ce_loss(
            logits_nar.unsqueeze(2),
            target_nar.unsqueeze(2), 
            dec_seq_lengths - 1
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
            "acc": (stats_ar[f"acc_layer0"] + stats_nar[f"acc_layer0"]) / 2,
            "weight": weight,
        }
        return loss, stats, weight

    def encode_ar(self, input_ar: torch.Tensor, cache: dict = {}):
        x_ar = self.emb(input_ar)

        x_ar = self.ar_pos_enc(x_ar, cache=cache)
        for layer in self.ar_decoders:
            x_ar = layer(x_ar, cache=cache)
        x_ar = self.ar_post_ln(x_ar)
        
        logits_ar = self.lm_head(x_ar)

        return logits_ar

    def encode_nar(
        self, 
        input_nar: torch.Tensor, 
        level_idx: torch.Tensor,
        dec_seq_lengths: torch.Tensor = None,
    ):
        if dec_seq_lengths is not None:
            mask = length_mask(dec_seq_lengths).unsqueeze(1)
        else:
            mask = None
        x_nar = self.emb(input_nar).sum(dim=2)

        x_nar = self.nar_pos_enc(x_nar)
        for layer in self.nar_decoders:
            x_nar = layer(x_nar, level_idx - 1, mask)
        x_nar = self.nar_post_ln(x_nar)

        logits_nar = self.lm_head(x_nar)

        return logits_nar

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
        _ = self.encode_ar(prefix[:, :-1, 0], cache=cache) # exclude modality start

        # (3) auto-regressive loop on first code layer
        # (3.1) prepare 
        minlen = int(prefix.size(1) * opts.minlenratio) if opts.minlenratio > 0 else 0
        maxlen = int(prefix.size(1) * opts.maxlenratio)
        if opts.search_algo == "teacher_force":
            assert suffix is not None
            minlen = suffix.size(1)
            maxlen = suffix.size(1)

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

            generated['token'].append(gen_tok)
            generated['score'].append(gen_score)

            if opts.search_algo == "teacher_force":
                prev_tok = suffix[:, step, 0].unsqueeze(1)
            else:
                prev_tok = gen_tok

            # (3.3) detect ended hypotheses.
            finish_idx = torch.where(
                torch.any(prev_tok.squeeze(1) == opts.eos),
                step,
                finish_idx,
            )

            if torch.all(torch.ge(finish_idx, 0)):
                finish_idx = finish_idx +  1 # index to length -> + 1
                logging.info(f"Early termination with sample lengths: {finish_idx.cpu().tolist()}")
                break

        # (3.4) finalize auto-regressive
        for hook in hooks:
            hook.remove()
        cache = {}
        
        gen_tokens_ar = torch.cat(generated['token'], dim=1).unsqueeze(2)
        gen_scores_ar = torch.cat(generated['score'], dim=1).unsqueeze(2)

        # (4) non-auto-regressive loop on remained code layers
        # (4.1) prepare
        if opts.search_algo == "teacher_force":
            prev_tok = torch.cat([prefix[:, 1:, 0], suffix[:, :, 0]], dim=1)
        else:
            prev_tok = torch.cat([prefix[:, 1:, 0], gen_tokens_ar[:, :, 0]], dim=1)
        prev_tok = prev_tok.unsqueeze(2)
        
        level_idx_th = torch.arange(1, opts.nq).long().to(opts.device)
        level_idx_th = level_idx_th.unsqueeze(1).expand(-1, opts.nbest)

        prefix_len = prefix.size(1) - 1 # <sos> excluded
        length = prefix_len + finish_idx

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

            # print(f'generated token: ', gen_tok.size(), gen_tok, gen_tok[:, prefix_len:])

            generated["token"].append(gen_tok[:, prefix_len:])
            generated["score"].append(gen_score[:, prefix_len:])

            if opts.search_algo == "teacher_force":
                prev_tok_layer = torch.cat([prefix[:, 1:, step], suffix[:, :, step]], dim=1)
            else:
                prev_tok_layer = torch.cat([prefix[:, 1:, step], gen_tok[:, prefix_len:]], dim=1)
            prev_tok = torch.cat([prev_tok, prev_tok_layer.unsqueeze(2)], dim=2)

        # (5) compose AR and NAR results
        gen_tokens_nar = torch.stack(generated["token"], dim=2)
        gen_scores_nar = torch.stack(generated["score"], dim=2)

        gen_tokens = torch.cat([gen_tokens_ar, gen_tokens_nar], dim=2)
        gen_scores = torch.cat([gen_scores_ar, gen_scores_nar], dim=2)

        gen_tokens_list, gen_scores_list = [], []
        for b in range(opts.nbest):
            gen_tokens_list.append(gen_tokens[b][:finish_idx[b] - 1])
            gen_scores_list.append(gen_scores[b][:finish_idx[b] - 1])
        
        return gen_tokens_list, gen_scores_list