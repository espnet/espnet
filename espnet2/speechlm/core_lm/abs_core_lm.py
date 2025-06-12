#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from abc import ABC, abstractmethod
from typing import Dict, Tuple, List
from espnet2.speechlm.inference_utils import AbsInferenceConfig
import torch


class AbsCoreLM(torch.nn.Module, ABC):
    """
    The abstract CoreLM class for SpeechLM, which is the major component of SpeechLM.
    """

    @abstractmethod
    def forward(
        self,
        dec_seq: torch.Tensor,
        loss_mask: torch.Tensor,
        conti_feats: list = List[Tuple],
    ) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        """Model forward

        Args:
            dec_seq (LongTensor): Batch of decoder sequences (B, T, nq).
            loss_mask (BoolTensor): bool mask for loss computing on dec_seq (B, T, nq).
            conti_feats: List[Tuple], the list continuous feature tuples:
              - feat: Any
              - modality: str
              - time_idx: int
              - duration: int

        Outputs:
            x (FloatTensor): Transformer out embeddings (B, T - 1, nq, D).
            targets (LongTensor): the revised training targets (B, T - 1, nq).
            loss_mask (BoolTensor): the bool mask for loss computing on targets.
        """
        raise NotImplementedError

    def inference(
        self,
        prefill: torch.Tensor,
        reference: torch.Tensor,
        config: AbsInferenceConfig,
    ):
        """Inference

        Args:
            prefill (LongTensor): The input prefill condition sequences (B, T, nq).
            reference (LongTensor): The groundtruth of this generation, only used
              in teacher-forcing (B, T, nq).
            enc_seq (LongTensor): Encoder token sequence (B, T_enc, nq).
            config: the AbsInferenceConfig object to specify the inference config.
        """
        raise NotImplementedError
