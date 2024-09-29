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
        prefix_len: torch.Tensor = None,
        conti_feats: Tuple = None,
    ) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        """ARDelayLM forward for training.

        This forward function is very similar to that of ARParallelLM only except the
        delay interleave pattern

        Args:
            dec_seq (LongTensor): Batch of decoder sequences (B, T, nq).
            prefix_len (LongTensor): Lengths of condition part in dec_seq (B,).
            conti_feats (dict or None): continuous features.
        """

        dec_seq_delay = self.delay_interleave(dec_seq=dec_seq)

        return super().forward(
            dec_seq=dec_seq_delay,
            prefix_len=prefix_len,
            conti_feats=conti_feats,
        )

    def delay_interleave(
        self,
        dec_seq: torch.Tensor,
        pad: int = 0,
    ):
        B, T, nq = dec_seq.size()
        retval = (
            torch.ones(
                (B, T + self.nq - 1, nq), dtype=dec_seq.dtype, device=dec_seq.device
            )
            * pad
        )

        for n in range(self.nq):
            retval[:, n : n + T, n] = dec_seq[:, :, n]

        return retval

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
        conti_feats = None,
        suffix: torch.Tensor = None,
    ):
        """Delay Architecture Inference.

        Args:
            prefix (LongTensor): Prefix part of dec_seq (B, T, nq).
            opts (SpeechLMInferenceOptions): inference options.
            conti_feats: continuous features.
            suffix (LongTensor): suffix part of dec_seq (B, T, nq),
                usually the target sequence for teacher-forcing.
        """

        # (1) initialization
        self.decoders.init()

        # (2) splice-interleave-split
        prefix = prefix.expand(opts.nbest, -1, -1)
        start = torch.Tensor([opts.start] + [0 for _ in range(prefix.size(2) - 1)])
        start = start.tile(opts.nbest, 1, 1).long().to(opts.device)
        suffix = suffix.expand(opts.nbest, -1, -1)
        full_seq_delay = self.delay_interleave(torch.cat([prefix, start, suffix], dim=1))
        prefix = full_seq_delay[:, :prefix.size(1)]
        suffix = full_seq_delay[:, prefix.size(1):]

        prefix_emb = self.emb(prefix).sum(dim=2)
        prefix_emb = install_continuous_features(prefix_emb, conti_feats * opts.nbest)
        _ = self.decoders(prefix_emb)

        # (3) auto-regressive loop
        # (3.1) AR initialization
        minlen = int(prefix.size(1) * opts.minlenratio) if opts.minlenratio > 0 else 0
        maxlen = (
            int(prefix.size(1) * opts.maxlenratio)
            if opts.minlenratio > 0
            else self.n_ctx
        )
        if opts.search_algo == "teacher_force":
            minlen = suffix.size(1) - 1
            maxlen = suffix.size(1) - 1
        if maxlen + prefix.size(1) > self.n_ctx:
            maxlen = self.n_ctx - prefix.size(1)
        logging.info(f"maxlen={maxlen}, minlen={minlen}, reflen={suffix.size(1)}")

        generated = {"token": [], "score": []}
        finish_idx = torch.Tensor([-1]).expand(opts.nbest).long().to(opts.device)
        prev_tok = start
        modality_index = start[:, 0, 0]
        mask = modality_index_to_mask(modality_index, opts)
        
        for step in range(maxlen):
            if step < self.nq:
                prev_tok = torch.cat(
                    [prev_tok[:, :, :step], suffix[:, step : step + 1, step:]], dim=2
                )

            # (3.2) AR model prediction
            prev_emb = self.emb(prev_tok).sum(dim=2)
            h = self.decoders(prev_emb)
            h = h.unsqueeze(2) + self.head_emb.weight.tile(1, 1, 1, 1)[:, :, : self.nq]
            logits = self.lm_head(h[:, :, :1])  # [B, 1, nq, V]
            if self.aux_lm_head is not None:
                aux_logits = self.aux_lm_head(h[:, :, 1:])
                # NOTE(Jinchuan) use small number but not -inf, otherwise it will cause confict with
                # modality mask.
                pad_aux_logits = torch.ones_like(logits).repeat(1, 1, aux_logits.size(2), 1) * -1e10
                pad_aux_logits[..., opts.aux_start: opts.aux_start + aux_logits.size(3)] = aux_logits
                logits = torch.cat([logits, pad_aux_logits], dim=2)

            gen_tok, gen_score = logits_to_tokens(
                logits,
                opts,
                mask,
                allow_eos=step >= minlen,
            )

            if opts.search_algo == "teacher_force":
                prev_tok = suffix[:, step + 1 : step + 2]
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

            # more "self.nq - 1" steps after all finish_idx becomes non-negative
            if finish_idx.min() >= 0 and step - finish_idx.max() >= self.nq - 1:
                break

            if step == maxlen - 1:
                logging.warning(
                    f"Some examples cannot finish in {maxlen} steps: {finish_idx}"
                    f"Consider increasing the maxlenratio"
                )

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

        valid_idx = finish_idx.ne(-1).nonzero(as_tuple=True)[0]
        if len(valid_idx) == 0:
            return [], []

        gen_token_seq = torch.cat(generated["token"], dim=1)[valid_idx]
        gen_score_seq = torch.cat(generated["score"], dim=1)[valid_idx]

        gen_token_seq = self.inverse_delay_interleave(gen_token_seq)
        gen_score_seq = self.inverse_delay_interleave(gen_score_seq)

        finish_idx = finish_idx[valid_idx]

        gen_tokens, gen_scores = [], []
        for b in range(len(valid_idx)):
            gen_tokens.append(gen_token_seq[b][: finish_idx[b]])
            gen_scores.append(gen_score_seq[b][: finish_idx[b]])
            assert not torch.any(gen_tokens[-1].eq(opts.eos))

        return gen_tokens, gen_scores
