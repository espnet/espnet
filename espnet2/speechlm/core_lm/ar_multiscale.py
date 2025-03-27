#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Implementation of UniAudio architecture: https://arxiv.org/abs/2310.00704

import logging
from typing import Dict, Tuple

import torch

from espnet2.speechlm.core_lm.abs_core_lm import AbsCoreLM, SpeechLMInferenceOptions
from espnet2.speechlm.module.builtin import TransformerDecoder
from espnet2.speechlm.net_utils import (
    install_continuous_features,
    logits_to_tokens,
    modality_index_to_mask,
)


class ARMultiScaleLM(AbsCoreLM):
    def __init__(
        self,
        transformer,
        vocab_size: int,
        aux_vocab_size: int,
        nq: int,
        share_emb: bool = False,
        # local transformer param
        qk_norm: bool = False,
        l_att_unit: int = 256,
        l_head: int = 2,
        l_layer: int = 4,
    ):
        """Initialize MultiScaleLM

        Args:
            transformer (torch.nn.Module): the Transformer body implementation
            vocab_size (int): Dimention of vocabulary.
            aux_vocab_size (int): the size of auxuliary tokens, usually for codec tokens.
            nq (int): Number of codes for each token / frame, usually for speech codec.
            share_emb (bool): If true, share the embedding and lm_head weight.

            qk_norm: (bool): If true, apply LayerNorm to q and k in atention.
            dropout: (float): dropout rate for attention layers.
            g_att_unit (int): Dimention of global Transformer attention.
            g_head (int): Number of heads in global Transformer attention.
            g_layer (int): Number of layers in global Transformer.
            l_att_unit (int): Dimention of local Transformer attention.
            l_head (int): Number of heads in local Transformer attention.
            l_layer (int): Number of layers in local Transformer.
            n_ctx (int): maximum context length of global Transformer.
        """
        super(ARMultiScaleLM, self).__init__()

        self.emb = torch.nn.Embedding(vocab_size, transformer.d_model)
        self.lm_head = torch.nn.Linear(transformer.d_model, vocab_size, bias=False)
        if share_emb:
            self.lm_head.weight = self.emb.weight

        if nq > 1 and aux_vocab_size > 0:
            self.aux_lm_head = torch.nn.Linear(
                transformer.d_model, aux_vocab_size, bias=False
            )
        else:
            self.aux_lm_head = None

        self.g_decoders = transformer
        # The local transforemr is always from builtin implementation
        self.l_decoders = TransformerDecoder(
            None,
            n_ctx=nq,
            n_state=l_att_unit,
            n_head=l_head,
            n_layer=l_layer,
            qk_norm=qk_norm,
        )

        self.placeholder = torch.nn.parameter.Parameter(
            torch.randn(l_att_unit, requires_grad=True)
        )

        if transformer.d_model != l_att_unit:
            self.g2l = torch.nn.Linear(transformer.d_model, l_att_unit)
            self.l2g = torch.nn.Linear(l_att_unit, transformer.d_model)
        else:
            self.g2l = torch.nn.Identity()
            self.l2g = torch.nn.Identity()

        if hasattr(self.g_decoders, "init_embeddings"):
            self.g_decoders.init_embeddings(self.emb, self.lm_head)

        self.nq = nq
        self.n_ctx = transformer.n_ctx

    def forward(
        self,
        dec_seq: torch.Tensor,
        prefix_len: torch.Tensor = None,
        conti_feats: Tuple = None,
    ) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        """Auto-Regresive MultiScale forward for training

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

        target = dec_seq[:, 1:]
        x = dec_seq[:, :-1]

        # input embedding
        x = self.emb(x).sum(dim=2)
        x = install_continuous_features(x, conti_feats)

        # global
        x = self.g2l(self.g_decoders(x))

        # local
        B, T, _ = x.size()
        placeholder = self.placeholder.tile(B, T, 1, 1)  # [B, T, 1, D]
        local_x = self.g2l(self.emb(target))
        local_x = torch.cat([placeholder, local_x], dim=2)[:, :, :-1]
        x = local_x + x.unsqueeze(2)  # [B, T, nq, D]

        x = x.flatten(0, 1)
        x = self.l_decoders(x).view(local_x.size())
        x = self.l2g(x)

        # lm_logits
        logits = self.lm_head(x[:, :, :1])
        aux_logits = self.aux_lm_head(x[:, :, 1:]) if self.nq > 1 else None

        return (logits, aux_logits), target

    @torch.no_grad()
    def inference(
        self,
        prefix: torch.Tensor,
        opts: SpeechLMInferenceOptions,
        conti_feats=None,
        suffix: torch.Tensor = None,
        inference_length: int = -1,
    ):
        """Auto-Regresive MultiScale Inference.

        Args:
            prefix (LongTensor): Prefix part of dec_seq (B, T, nq).
            opts (SpeechLMInferenceOptions): inference options.
            conti_feats: continuous features.
            suffix (LongTensor): suffix part of dec_seq (B, T, nq),
                usually the target sequence for teacher-forcing.
            inference_length: a given inference length, ignore if -1
        """

        # (1) initialization
        self.g_decoders.init()

        # (2) Prefix forward
        prefix = prefix.expand(opts.nbest, -1, -1)
        suffix = suffix.expand(opts.nbest, -1, -1)
        prefix_emb = self.emb(prefix).sum(2)
        _ = self.g_decoders(prefix_emb)

        # (3) global loop
        # (3.1) global initialization
        minlen = int(prefix.size(1) * opts.minlenratio) if opts.minlenratio > 0 else 0
        maxlen = (
            int(prefix.size(1) * opts.maxlenratio)
            if opts.minlenratio > 0
            else self.n_ctx - prefix.size(1)
        )

        if opts.fixed_length:
            assert (
                inference_length > 0
            ), "Inference length is needed for fixed length inference"
            minlen = int(inference_length)
            maxlen = int(inference_length)

        if opts.search_algo == "teacher_force":
            minlen = suffix.size(1)
            maxlen = suffix.size(1)

        logging.info(f"maxlen={maxlen}, minlen={minlen}, reflen={suffix.size(1)}")

        finish_idx = torch.Tensor([-1]).expand(opts.nbest).long().to(opts.device)
        g_generated = {"token": [], "score": []}
        g_prev_tok = torch.Tensor([opts.start] + [0 for _ in range(prefix.size(2) - 1)])
        g_prev_tok = g_prev_tok.tile(opts.nbest, 1, 1).long().to(opts.device)
        modality_index = g_prev_tok[:, 0, 0]
        mask = modality_index_to_mask(modality_index, opts)

        for g_step in range(1, maxlen + 1):
            g_prev_emb = self.emb(g_prev_tok).sum(2)  # [B, 1, D]
            g_hidden = self.g2l(self.g_decoders(g_prev_emb))  # [B, 1, D]

            # (3.2) local initialization
            self.l_decoders.init()

            # (3.3) local loop
            l_generated = {"token": [], "score": []}
            l_prev_emb = self.placeholder.tile(opts.nbest, 1, 1)  # [B, 1, D]
            for l_step in range(opts.nq):
                # print(f"local addition: {l_step} | {l_prev_emb[:, 0, 0]} {g_hidden[:, 0, 0]}")
                l_hidden = l_prev_emb + g_hidden
                l_hidden = self.l2g(self.l_decoders(l_hidden))  # [B, 1, D]

                if l_step == 0:
                    logits = self.lm_head(l_hidden)
                else:
                    logits = self.aux_lm_head(l_hidden)
                    logits = torch.nn.functional.pad(
                        logits,
                        pad=(
                            opts.aux_start,
                            self.lm_head.out_features
                            - (opts.aux_start + logits.size(2)),
                            0,
                            0,
                            0,
                            0,
                        ),
                        mode="constant",
                        value=-1e10,
                    )

                gen_tok, gen_score = logits_to_tokens(
                    logits.unsqueeze(2),
                    opts,
                    mask,
                    allow_eos=(l_step == 0 and g_step >= minlen),
                    nq_level=l_step,
                )
                # [B, 1, 1] -> [B, 1]
                gen_tok, gen_score = gen_tok.squeeze(2), gen_score.squeeze(2)

                if opts.search_algo == "teacher_force":
                    l_prev_tok = suffix[:, g_step - 1 : g_step, l_step]
                else:
                    l_prev_tok = gen_tok
                # print(f"gen_tok: {gen_tok} | l_prev_tok: {l_prev_tok}", flush=True)
                l_prev_emb = self.g2l(self.emb(l_prev_tok))

                l_generated["token"].append(gen_tok)
                l_generated["score"].append(gen_score)

            # (3.4) local finalize
            self.l_decoders.reset()

            gen_tokens_local = torch.stack(l_generated["token"], dim=2)  # [B, 1, nq]
            gen_scores_local = torch.stack(l_generated["score"], dim=2)

            g_generated["token"].append(gen_tokens_local)
            g_generated["score"].append(gen_scores_local)

            if opts.search_algo == "teacher_force":
                g_prev_tok = suffix[:, g_step - 1 : g_step]
            else:
                g_prev_tok = gen_tokens_local

            # (3.5) detect ended hypotheses
            finish_idx = torch.where(
                torch.logical_and(g_prev_tok[:, 0, 0] == opts.eos, finish_idx == -1),
                g_step,
                finish_idx,
            )

            if torch.all(torch.ge(finish_idx, 0)):
                break

            if g_step == maxlen and torch.any(finish_idx == -1):
                logging.warning(
                    f"Some examples cannot finish in {maxlen} steps: {finish_idx} "
                    f"Force it to finish"
                )
                torch.where(finish_idx == -1, g_step, finish_idx)

            # (3.6) detect modality switch
            modality_change_mask = torch.logical_and(
                g_prev_tok[:, 0, 0] >= 32,
                g_prev_tok[:, 0, 0] < 64,
            )
            if torch.any(modality_change_mask):
                modality_index = torch.where(
                    modality_change_mask,
                    g_prev_tok[:, 0, 0],
                    modality_index,
                )
                mask = modality_index_to_mask(modality_index, opts)
                logging.warning(
                    f"Step {g_step}: change modality index {modality_index}"
                )

        logging.info(f"Finish with lengths: {finish_idx.cpu().tolist()}")

        # (4) global finalize & build hypotheses
        self.g_decoders.reset()

        g_generated = {
            "token": torch.cat(g_generated["token"], dim=1),
            "score": torch.cat(g_generated["score"], dim=1),
        }

        gen_tokens, gen_scores = [], []
        for b in range(len(finish_idx)):
            # -1 to exclude eos
            gen_tokens.append(g_generated["token"][b][: finish_idx[b] - 1])
            gen_scores.append(g_generated["score"][b][: finish_idx[b] - 1])
            assert not torch.any(gen_tokens[-1].eq(opts.eos))

        return gen_tokens, gen_scores
