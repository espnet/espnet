#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Duration calculator related modules."""

import torch

from espnet.nets.pytorch_backend.e2e_tts_transformer import Transformer
from espnet.nets.pytorch_backend.nets_utils import pad_list


class DurationCalculator(torch.nn.Module):
    """Duration calculator module for FastSpeech.

    Todo:
        * Fix the duplicated calculation of diagonal head decision

    """

    def __init__(self, teacher_model):
        """Initialize duration calculator module.

        Args:
            teacher_model (e2e_tts_transformer.Transformer): Pretrained auto-regressive Transformer.

        """
        super(DurationCalculator, self).__init__()
        if not isinstance(teacher_model, Transformer):
            raise ValueError("teacher model should be the instance of e2e_tts_transformer.Transformer")
        self.teacher_model = teacher_model
        self.register_buffer("diag_head_idx", torch.tensor(-1))

    def forward(self, xs, ilens, ys, olens, spembs=None):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of the padded sequences of character ids (B, Tmax).
            ilens (Tensor): Batch of lengths of each input sequence (B,).
            ys (Tensor): Batch of the padded sequence of target features (B, Lmax, odim).
            olens (Tensor): Batch of lengths of each output sequence (B,).
            spembs (Tensor, optional): Batch of speaker embedding vectors (B, spk_embed_dim).

        Returns:
            Tensor: Batch of durations (B, Tmax).

        """
        att_ws = self._calculate_encoder_decoder_attentions(xs, ilens, ys, olens, spembs=spembs)
        # TODO(kan-bayashi): fix this issue
        # this does not work in multi-gpu case. registered buffer is not saved.
        if int(self.diag_head_idx) == -1:
            self._init_diagonal_head(att_ws)
        att_ws = att_ws[:, self.diag_head_idx]
        durations = [self._calculate_duration(att_w, ilen, olen) for att_w, ilen, olen in zip(att_ws, ilens, olens)]

        return pad_list(durations, 0)

    @staticmethod
    def _calculate_duration(att_w, ilen, olen):
        return torch.stack([att_w[:olen, :ilen].argmax(-1).eq(i).sum() for i in range(ilen)])

    def _init_diagonal_head(self, att_ws):
        diagonal_scores = att_ws.max(dim=-1)[0].mean(dim=-1).mean(dim=0)  # (H * L,)
        self.register_buffer("diag_head_idx", diagonal_scores.argmax())

    def _calculate_encoder_decoder_attentions(self, xs, ilens, ys, olens, spembs=None):
        att_dict = self.teacher_model.calculate_all_attentions(
            xs, ilens, ys, olens, spembs=spembs, skip_output=True, keep_tensor=True)
        return torch.cat([att_dict[k] for k in att_dict.keys() if "src_attn" in k], dim=1)  # (B, H*L, Lmax, Tmax)
