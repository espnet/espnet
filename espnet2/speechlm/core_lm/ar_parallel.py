#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Implementation of Parallel architecture: https://arxiv.org/pdf/2306.05284

from typing import Dict, List, Tuple

import torch

from espnet2.speechlm.core_lm.abs_core_lm import AbsCoreLM, SpeechLMInferenceOptions
from espnet2.speechlm.net_utils import install_continuous_features


class ARParallelLM(AbsCoreLM):
    def __init__(
        self,
        transformer,
        continuous_encoders: dict,
        vocab_size: int,
        nq: int,
        share_emb: bool = False,
    ):
        """Initialize Auto-regressive LM with parallel interleave codec pattern.

        Args:
            transformer (torch.nn.Module): the Transformer body implementation
            vocab_size (int): Dimention of vocabulary.
            nq (int): Number of codes for each token / frame, usually for speech codec.
            share_emb (bool): If true, share the embedding and lm_head weight.
        """
        super(ARParallelLM, self).__init__()

        self.decoders = transformer

        self.emb = torch.nn.Embedding(vocab_size, transformer.d_model, padding_idx=0)

        self.lm_head = torch.nn.Linear(transformer.d_model, vocab_size, bias=False)
        if share_emb:
            self.lm_head.weight = self.emb.weight

        self.head_emb = torch.nn.Embedding(12, transformer.d_model, padding_idx=0)
        # self.head_emb.weight[0] = 0

        if hasattr(self.decoders, "init_embeddings"):
            self.decoders.init_embeddings(self.emb, self.lm_head)

        self.continuous_encoders = torch.nn.ModuleDict(continuous_encoders)
        for _, module in continuous_encoders.items():
            module.register_projection(odim=transformer.d_model)

        self.nq = nq
        self.n_ctx = transformer.n_ctx

    def forward(
        self,
        dec_seq: torch.Tensor,
        loss_mask: torch.Tensor = None,
        conti_feats: list = None,
    ) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        """Auto-Regresive LM forward for training

        Args:
            dec_seq (LongTensor): Batch of decoder sequences (B, T, nq).
            loss_mask (LongTensor): Lengths of condition part in dec_seq (B, T, nq).
        """
        assert dec_seq.dim() == 3

        targets = dec_seq[:, 1:]
        loss_mask = loss_mask[:, 1:] if loss_mask is not None else None
        x = dec_seq[:, :-1]

        # input embedding
        x = self.emb(x).sum(dim=2)
        if len(self.continuous_encoders) > 0:
            for modality, module in self.continuous_encoders.items():
                x = module(
                    x,
                    conti_feats,
                    modality=modality,
                )

        # transformer output
        x = self.decoders(x)
        x = x.unsqueeze(2) + self.head_emb.weight.tile(1, 1, 1, 1)[:, :, : self.nq]

        # NOTE(Jinchuan): We don't apply lm_head here naively. It is implemented in
        # loss module to save computing.
        return x, targets, loss_mask

    @torch.no_grad()
    def inference(
        self,
        prefix: torch.Tensor,
        opts: SpeechLMInferenceOptions,
        conti_feats=None,
        suffix: torch.Tensor = None,
    ):
        """Auto-Regresive MultiScale Inference.

        Args:
            prefix (LongTensor): Prefix part of dec_seq (B, T_dec, nq).
            opts (SpeechLMInferenceOptions): inference options.
            conti_feats: continuous features.
            suffix (LongTensor): suffix part of dec_seq (B, T_dec, nq),
                usually the target sequence for teacher-forcing.
        """

        raise NotImplementedError
