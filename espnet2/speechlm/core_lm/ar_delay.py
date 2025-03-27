#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Implementation of Delay architecture: https://arxiv.org/pdf/2306.05284

import logging
from typing import Dict, Tuple

import torch

from espnet2.speechlm.core_lm.abs_core_lm import SpeechLMInferenceOptions
from espnet2.speechlm.core_lm.ar_parallel import ARParallelLM
from espnet2.speechlm.net_utils import (
    logits_to_tokens,
    modality_index_to_mask,
    install_continuous_features,
)


class ARDelayLM(ARParallelLM):
    def forward(
        self,
        dec_seq: torch.Tensor,
        loss_mask: torch.Tensor = None,
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
        prefix: torch.Tensor,
        opts: SpeechLMInferenceOptions,
        conti_feats=None,
        suffix: torch.Tensor = None,
        inference_length: int = -1,
    ):
        """Delay Architecture Inference.

        Args:
            prefix (LongTensor): Prefix part of dec_seq (B, T, nq).
            opts (SpeechLMInferenceOptions): inference options.
            conti_feats: continuous features.
            suffix (LongTensor): suffix part of dec_seq (B, T, nq),
                usually the target sequence for teacher-forcing.
            inference_length: a given inference length, ignore if -1
        """

        # (1) initialization
        self.decoders.init()

        # (2) splice-interleave-split and prefill
        full_seq_delay, _ = self.delay_interleave(torch.cat([prefix, suffix], dim=1))
        full_seq_delay = full_seq_delay.expand(opts.nbest, -1, -1)

        prelen = prefix.size(1)
        prefix = full_seq_delay[:, : prelen - 1]
        prev_tok = full_seq_delay[:, prelen - 1 : prelen]
        suffix = full_seq_delay[:, prelen - 1 :]

        prefix_emb = self.emb(prefix).sum(dim=2)
        _ = self.decoders(prefix_emb)

        # (3) auto-regressive loop
        # (3.1) AR initialization

        # NOTE(Jinchuan):
        # The delay interleave will need more "self.nq - 1" infernece step to obtain
        # the effective codes in each level. minlen and maxlin already count them.
        minlen = int(prefix.size(1) * opts.minlenratio) if opts.minlenratio > 0 else 0
        maxlen = (
            int(prefix.size(1) * opts.maxlenratio) + (self.nq - 1)
            if opts.maxlenratio > 0
            else self.n_ctx - prefix.size(1)
        )

        if opts.fixed_length:
            assert (
                inference_length > 0
            ), "Inference length is needed for fixed length inference"
            minlen = int(inference_length) + (self.nq - 1)
            maxlen = int(inference_length) + (self.nq - 1)

        if opts.search_algo == "teacher_force":
            minlen = suffix.size(1)
            maxlen = suffix.size(1)

        logging.info(f"maxlen={maxlen}, minlen={minlen}, reflen={suffix.size(1)}")

        generated = {"token": [], "score": []}
        finish_idx = torch.Tensor([-1]).expand(opts.nbest).long().to(opts.device)
        # initially, modality is unknown
        mask = opts.masks["unknown"].tile(opts.nbest, 1, 1, 1)
        modality_index = prev_tok[:, :, 0].flatten() * 0

        for step in range(1, maxlen + 1):
            # (3.2) AR model prediction
            prev_emb = self.emb(prev_tok).sum(dim=2)
            h = self.decoders(prev_emb)

            h = h.unsqueeze(2) + self.head_emb.weight.tile(1, 1, 1, 1)[:, :, : self.nq]
            logits = self.lm_head(h[:, :, :1])  # [B, 1, nq, V]
            if self.aux_lm_head is not None:
                aux_logits = self.aux_lm_head(h[:, :, 1:])
                # NOTE(Jinchuan) use small number but not -inf, otherwise it will cause confict with
                # modality mask.
                pad_aux_logits = (
                    torch.ones_like(logits).repeat(1, 1, aux_logits.size(2), 1) * -1e10
                )
                pad_aux_logits[
                    ..., opts.aux_start : opts.aux_start + aux_logits.size(3)
                ] = aux_logits
                logits = torch.cat([logits, pad_aux_logits], dim=2)

            gen_tok, gen_score = logits_to_tokens(
                logits, opts, mask, allow_eos=step >= minlen - (self.nq - 1)
            )

            # NOTE(Jinchuan): Due to delay interleave, the first predictions
            # are replaced by PAD. Especially, when step == 1, the prediction
            # is the modality identifier.
            if step < self.nq + 1:
                pad_start = max(1, step - 1)
                gen_tok[:, :, pad_start:] = 0

            if opts.search_algo == "teacher_force":
                prev_tok = suffix[:, step : step + 1]
            else:
                prev_tok = gen_tok

            generated["token"].append(gen_tok)
            generated["score"].append(gen_score)

            # (3.3) detect ended hypotheses
            finish_idx = torch.where(
                torch.logical_and(prev_tok[:, 0, 0] == opts.eos, finish_idx == -1),
                step,
                finish_idx,
            )

            if step == maxlen - (self.nq - 1) and torch.any(finish_idx == -1):
                logging.warning(
                    f"Some examples cannot finish in {maxlen} steps: {finish_idx} "
                    f"Force it to finish. "
                )
                finish_idx = torch.where(finish_idx == -1, step, finish_idx)

            # more "self.nq - 1" steps after all finish_idx becomes non-negative
            if finish_idx.min() >= 0 and step - finish_idx.max() >= self.nq - 1:
                break

            # (3.4) detect modality swtich
            modality_change_mask = torch.logical_and(
                prev_tok[:, 0, 0] >= 32,
                prev_tok[:, 0, 0] < 64,
            )
            if torch.any(modality_change_mask):
                modality_index = torch.where(
                    modality_change_mask,
                    prev_tok[:, 0, 0],
                    modality_index,
                )
                mask = modality_index_to_mask(modality_index, opts)
                logging.warning(f"Step {step}: change modality index {modality_index}")

        logging.info(f"Finish with lengths: {finish_idx.cpu().tolist()}")

        # (4) global finalize & build hypotheses
        self.decoders.reset()

        gen_token_seq = torch.cat(generated["token"], dim=1)
        gen_score_seq = torch.cat(generated["score"], dim=1)

        gen_token_seq = self.inverse_delay_interleave(gen_token_seq)
        gen_score_seq = self.inverse_delay_interleave(gen_score_seq)

        gen_tokens, gen_scores = [], []
        for b in range(len(finish_idx)):
            gen_tokens.append(gen_token_seq[b][: finish_idx[b] - 1])
            gen_scores.append(gen_score_seq[b][: finish_idx[b] - 1])
            assert not torch.any(gen_tokens[-1].eq(opts.eos))

        return gen_tokens, gen_scores
