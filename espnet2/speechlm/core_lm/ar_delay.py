#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Implementation of UniAudio architecture: https://arxiv.org/abs/2310.00704

import logging
from typing import Dict, Tuple

import torch

from espnet2.speechlm.core_lm.abs_core_lm import SpeechLMInferenceOptions
from espnet2.speechlm.core_lm.ar_parallel import ARParallelLM


class ARDelayLM(ARParallelLM):
    def forward(
        self,
        dec_seq: torch.Tensor,
        dec_seq_lengths: torch.Tensor = None,
        enc_seq: torch.Tensor = None,
        enc_seq_lengths: torch.Tensor = None,
        prefix_len: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        """ARDelayLM forward for training.
        This forward function is very similar to that of ARParallelLM only except the 
        delay interleave pattern

        Args:
            dec_seq (LongTensor): Batch of decoder sequences (B, T, nq).
            dec_seq_lengths (LongTensor): Lengths of batched decoder sequences (B,).
            enc_seq (LongTensor): Batch of encoder sequences (B, T, nq), keep the interface,
                may not be used.
            enc_seq_lengths (LongTensor): Lengths of batched encoder sequences (B,),
                keep the interface, may not be used.
            prefix_len (LongTensor): Lengths of condition part in dec_seq (B,).
        """
        
        B, T, nq = dec_seq.size()
        dec_seq_delay = torch.ones(
            (B, T + nq - 1, nq),
            dtype=dec_seq.dtype,
            device=dec_seq.device,
        ) * self.sos_eos

        for n in range(nq):
            dec_seq_delay[:, n: n+T, n] = dec_seq[:, :, n]
        
        dec_seq_lengths_delay = dec_seq_lengths + nq - 1

        return super().forward(
            dec_seq=dec_seq_delay,
            dec_seq_lengths=dec_seq_lengths_delay,
            enc_seq=enc_seq,
            enc_seq_lengths=enc_seq_lengths,
            prefix_len=prefix_len,
        )

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

        raise NotImplementedError
    
    def delay_interleave(
        self,
        dec_seq: torch.Tensor,
        dec_seq_lengths: torch.Tensor,
        pad: int = 0,
    ):
        B, T, nq = dec_seq.size()
        retval = torch.ones(
            (B, T + self.nq - 1, nq), 
            dtype=dec_seq.dtype, 
            device=dec_seq.device
        ) * pad

        for n in range(self.nq):
            retval[:, n: n + T, n] = dec_seq[:, :, n]
        
        dec_seq_lengths = dec_seq_lengths + self.nq - 1

        return retval, dec_seq_lengths

