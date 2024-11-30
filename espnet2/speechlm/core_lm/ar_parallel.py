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
        vocab_size: int,
        aux_vocab_size: int,
        nq: int,
        share_emb: bool = False,
    ):
        """Initialize Auto-regressive LM with parallel interleave codec pattern.

        Args:
            transformer (torch.nn.Module): the Transformer body implementation
            vocab_size (int): Dimention of vocabulary.
            aux_vocab_size (int): the size of auxuliary tokens, usually for codec tokens.
            nq (int): Number of codes for each token / frame, usually for speech codec.
            share_emb (bool): If true, share the embedding and lm_head weight.
        """
        super(ARParallelLM, self).__init__()

        self.decoders = transformer

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
        self.head_emb = torch.nn.Embedding(12, transformer.d_model) 

        if hasattr(self.decoders, "init_embeddings"):
            self.decoders.init_embeddings(self.emb, self.lm_head)
        
        self.nq = nq
        self.n_ctx = transformer.n_ctx
        
    def forward(
        self,
        dec_seq: torch.Tensor,
        prefix_len: torch.Tensor = None,
        pos_id: torch.Tensor = None,
        conti_feats: Tuple = None,
    ) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        """Auto-Regresive LM forward for training

        Args:
            dec_seq (LongTensor): Batch of decoder sequences (B, T, nq).
            prefix_len (LongTensor): Lengths of condition part in dec_seq (B,).
            conti_feats (dict or None): continuous features.
        """
        assert dec_seq.dim() == 3

        target = dec_seq[:, 1:]
        x = dec_seq[:, :-1]

        # input embedding
        x = self.emb(x).sum(dim=2)
        x = install_continuous_features(x, conti_feats)

        # transformer output
        x = self.decoders(x, pos_id=pos_id)
        assert 1 == 2
        x = x.unsqueeze(2) + self.head_emb.weight.tile(1, 1, 1, 1)[:, :, : self.nq]

        # lm logits
        logits = self.lm_head(x[:, :, :1])
        aux_logits = self.aux_lm_head(x[:, :, 1:]) if self.nq > 1 else None

        return (logits, aux_logits), target

    @torch.no_grad()
    def inference(
        self,
        prefix: torch.Tensor,
        opts: SpeechLMInferenceOptions,
        conti_feats = None,
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
