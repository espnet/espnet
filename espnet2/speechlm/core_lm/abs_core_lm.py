#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass
class SpeechLMInferenceOptions:
    device: str = "cpu"
    search_algo: str = "sampling"
    nbest: int = 1
    sampling_temperature: float = 1.0
    top_k: int = 20
    maxlenratio: float = 0.0
    minlenratio: float = 0.0
    eos: int = 5
    start: int = 1
    masks: torch.Tensor = None
    nq: int = None


class AbsCoreLM(torch.nn.Module, ABC):
    """The abstract CoreLM class for SpeechLM, which is the major component of SpeechLM.

    It supports or is going to support several styles of SpeechLM:
    Auto-Regressive (AR):
          SpearTTS: https://arxiv.org/abs/2302.03540 (TODO)
          MusicGen: https://arxiv.org/abs/2306.05284 (TODO)
          UniAudio: https://arxiv.org/abs/2310.00704

    Non-Auto-Regressive (NAR):
          SoundStorm: https://arxiv.org/abs/2305.09636 (TODO)

    Auto-Regressive + Non-Auto-Regressive (AR + NRA): Hybrid of AR and NAR.
          Vall-E: https://arxiv.org/abs/2301.02111

    For developers: to build a new core_lm model, try to follow:
        (1) Build with Espnet Espnet internal modules:
            Use modules from `espnet2.speechlm.module.transformer.py`. If you get
            some modules that is specific to your model, put them under
            `espnet2.speechlm.module.<model_name>.py`.
        (2) or, Build with HuggingFace model/modules:
            Put everyhing in `espnet2.speechlm.core_lm.<model_name>.py`. Usually
            this is just a warpper that bridges HF models into Espnet SpeechLM.
    Reminder: try to avoid any model dependency beyond espnet2.speechlm
    """

    @abstractmethod
    def forward(
        self,
        dec_seq: torch.Tensor,
        dec_seq_lengths: torch.Tensor = None,
        enc_seq: torch.Tensor = None,
        enc_seq_lengths: torch.Tensor = None,
        prefix_len: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        """Model forward

        Args:
            dec_seq (LongTensor): Batch of decoder sequences (B, T, nq).
            dec_seq_lengths (LongTensor): Lengths of batched decoder sequences (B,).
            enc_seq (LongTensor): Batch of encoder sequences (B, T, nq), keep
                the interface, may not be used.
            enc_seq_lengths (LongTensor): Lengths of batched encoder sequences (B,),
                keep the interface, may not be used.
            prefix_len (LongTensor): Lengths of condition part in dec_seq (B,).
        """
        raise NotImplementedError

    def inference(
        self,
        prefix: torch.Tensor,
        opts: SpeechLMInferenceOptions,
        enc_seq: torch.Tensor = None,
        suffix: torch.Tensor = None,
    ):
        """Inference

        Args:
            prefix (LongTensor): Prefix part of dec_seq (B, T_dec, nq).
            opts (SpeechLMInferenceOptions): inference options.
            enc_seq (LongTensor): Encoder token sequence (B, T_enc, nq).
            suffix (LongTensor): suffix part of dec_seq (B, T_dec, nq),
                usually the target sequence for teacher-forcing.
        """
        raise NotImplementedError
