#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Implementation of Delay architecture: https://arxiv.org/pdf/2306.05284

import logging
from typing import Dict, Tuple

import torch

from espnet2.speechlm.core_lm.abs_core_lm import SpeechLMInferenceOptions
from espnet2.speechlm.core_lm.ar_parallel import ARParallelLM
from espnet2.speechlm.inference_utils import AbsInferenceConfig
from espnet2.speechlm.net_utils import (
    install_continuous_features,
    logits_to_tokens,
    modality_index_to_mask,
)


class ARDelayLM(ARParallelLM):
    def forward(
        self,
        dec_seq: torch.Tensor,
        loss_mask: torch.Tensor = None,
        conti_feats: list = None,
    ) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        """ARDelayLM forward for training.

        This forward function is very similar to that of ARParallelLM only except the
        delay interleave pattern

        Args:
            dec_seq (LongTensor): Batch of decoder sequences (B, T, nq).
            loss_mask (LongTensor): Lengths of condition part in dec_seq (B, T, nq).
        """
        dec_seq_delay, loss_mask = self.delay_interleave(
            dec_seq=dec_seq, loss_mask=loss_mask
        )

        return super().forward(
            dec_seq=dec_seq_delay,
            loss_mask=loss_mask,
            conti_feats=conti_feats,
        )

    def delay_interleave(
        self,
        dec_seq: torch.Tensor,
        loss_mask: torch.Tensor = None,
        pad: int = 0,
    ):
        B, T, nq = dec_seq.size()
        retval = (
            torch.ones(
                (B, T + self.nq - 1, nq), dtype=dec_seq.dtype, device=dec_seq.device
            )
            * pad
        )
        ret_loss_mask = retval.clone().detach() if loss_mask is not None else None

        for n in range(self.nq):
            retval[:, n : n + T, n] = dec_seq[:, :, n]
            if ret_loss_mask is not None:
                ret_loss_mask[:, n : n + T, n] = loss_mask[:, :, n]

        retval = retval[:, : -(self.nq - 1)]
        if ret_loss_mask is not None:
            ret_loss_mask = ret_loss_mask[:, : -(self.nq - 1)]

        return retval, ret_loss_mask

    def inverse_delay_interleave(
        self,
        dec_seq_delay: torch.Tensor,
    ):
        retval = []
        length = dec_seq_delay.size(1) - self.nq + 1
        for i in range(dec_seq_delay.size(2)):
            retval.append(dec_seq_delay[:, i : i + length, i])

        retval = torch.stack(retval, dim=2)

        return retval

    @torch.no_grad()
    def inference(
        self,
        prefill: torch.Tensor,
        reference: torch.Tensor,
        config: AbsInferenceConfig,
    ):
        # (1) Prefill
        prefill_delay, _ = self.delay_interleave(prefill)
        if config.search_algo == "teacher_force":
            reference_delay, _ = self.delay_interleave(reference)
        else:
            reference_delay = None

        prefill_delay_emb = self.emb(prefill_delay[:, :-1]).sum(dim=2)
        _ = self.decoders(prefill_delay_emb)

        # (2) Length control
        # TODO(Jinchuan): double-check the length control logic
        if config.search_algo == "teacher_force":
            maxlen = reference_delay.size(1)
            minlen = reference_delay.size(1)
        elif config.length_method == "absolute":
            maxlen = config.maxlen
            minlen = config.minlen
        elif config.length_method == "relative":
            raise NotImplementedError

        # (3) auto-regressive loop
        # (3.1) AR initialization
        logging.info(f"maxlen={maxlen}, minlen={minlen}")
        if reference_delay is not None:
            logging.info(f"Using teacher force, ref. length={reference_delay.size(1)}")

        generated = {"token": [], "score": []}
        finish_idx = torch.ones(config.nbest) * -1
        finish_idx = finish_idx.long().to(config.device)
        nq_axis = torch.arange(config.nq).long().to(config.device)
        prev_tok = prefill_delay[:, -1:]

        for step in range(0, maxlen):
            # (3.2) AR model prediction
            prev_emb = self.emb(prev_tok).sum(dim=2)
            h = self.decoders(prev_emb)
            h = h.unsqueeze(2) + self.head_emb.weight.tile(1, 1, 1, 1)[:, :, : self.nq]
            logits = self.lm_head(h)

            gen_tok, gen_score = logits_to_tokens(
                logits,
                config,
                allow_eos=step >= minlen,
            )

            # NOTE(Jinchuan): Force some tokens to be PAD (0) due to delay interleave
            if step <= self.nq:
                gen_tok[:, :, step + 1 :] = 0

            if torch.any(finish_idx != -1):
                finish_step = torch.where(finish_idx > 0, step - finish_idx + 1, 0)
                gen_tok = torch.where(
                    finish_step.view(-1, 1, 1) > nq_axis.view(1, 1, -1), 0, gen_tok
                )

            # (3.3) detect ended hypotheses
            finish_idx = torch.where(
                torch.logical_and(prev_tok[:, 0, 0] == config.eos, finish_idx == -1),
                step,
                finish_idx,
            )

            if step == maxlen - (self.nq - 1) and torch.any(finish_idx == -1):
                logging.warning(
                    f"Some examples cannot finish in {maxlen} steps: {finish_idx} "
                    f"Force it to finish. "
                )
                gen_tok[:, 0, 0] = torch.where(
                    finish_idx == -1, config.eos, gen_tok[:, 0, 0]
                )
                finish_idx = torch.where(finish_idx == -1, step, finish_idx)

            if config.search_algo == "teacher_force":
                prev_tok = reference_delay[:, step : step + 1]
            else:
                prev_tok = gen_tok

            generated["token"].append(gen_tok)
            generated["score"].append(gen_score)

            # more "self.nq - 1" steps after all finish_idx becomes non-negative
            if finish_idx.min() >= 0 and step - finish_idx.max() >= self.nq - 1:
                break

        # (3.4) forward the last `prev_tok` to keep the correct KV-Cache
        prev_emb = self.emb(prev_tok).sum(dim=2)
        h = self.decoders(prev_emb)

        logging.info(f"Finish with lengths: {finish_idx.cpu().tolist()}")

        gen_token_seq = torch.cat(generated["token"], dim=1)
        gen_score_seq = torch.cat(generated["score"], dim=1)

        gen_token_seq = self.inverse_delay_interleave(gen_token_seq)
        gen_score_seq = self.inverse_delay_interleave(gen_score_seq)

        gen_tokens, gen_scores = [], []
        for b in range(len(finish_idx)):
            gen_tokens.append(gen_token_seq[b][: finish_idx[b] - 1])
            gen_scores.append(gen_score_seq[b][: finish_idx[b] - 1])
            assert not torch.any(gen_tokens[-1].eq(config.eos))

        return gen_tokens, gen_scores
