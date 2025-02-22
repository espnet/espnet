#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from typing import Dict, Tuple

import torch
import torch.nn.functional as F  # noqa
from typeguard import typechecked

from espnet2.speechlm.core_lm.abs_core_lm import AbsCoreLM
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel


@typechecked
class ESPnetSpeechLMModel(AbsESPnetModel):
    def __init__(
        self,
        corelm: AbsCoreLM,
        extract_feats_in_collect_stats: bool = False,
    ):
        super().__init__()

        self.corelm = corelm
        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    def forward(
        self,
        dec_seq: torch.Tensor,
        dec_seq_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:

        enc_seq = kwargs.get("enc_seq", None)
        enc_seq_lengths = kwargs.get("enc_seq_lengths", None)
        prefix_len = kwargs.get("prefix_len", None)
        if prefix_len is not None:
            prefix_len = prefix_len.squeeze(1)

        loss, stats, weight = self.corelm(
            dec_seq,
            dec_seq_lengths,
            enc_seq,
            enc_seq_lengths,
            prefix_len,
        )

        loss, stats, weight = force_gatherable((loss, stats, weight), loss.device)
        return loss, stats, weight

    def collect_feats(self, **kwargs):
        raise NotImplementedError
