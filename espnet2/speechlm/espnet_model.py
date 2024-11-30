#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from typing import Dict, Tuple, Optional

import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.speechlm.core_lm.abs_core_lm import AbsCoreLM
from espnet2.speechlm.module.builtin import ResidualAttentionBlock
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel


@typechecked
class ESPnetSpeechLMModel(AbsESPnetModel):

    @typechecked
    def __init__(
        self,
        corelm: AbsCoreLM,
        criterion,
        extract_feats_in_collect_stats: bool = False,
    ):
        super().__init__()

        self.corelm = corelm
        self.criterion = criterion
        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    def forward(
        self,
        dec_seq: torch.Tensor,
        dec_seq_lengths: torch.Tensor,
        prefix_len: Optional[torch.Tensor] = None,
        conti_feats = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        prefix_len = prefix_len.squeeze(1) if prefix_len is not None else None
        pos_id = kwargs.get("pos_id", None)

        logits, targets = self.corelm(
            dec_seq,
            prefix_len,
            pos_id,
            conti_feats,
        )

        loss, stats, weight = self.criterion(
            logits, 
            targets, 
            prefix_len,
            dec_seq_lengths,
        )
        loss, stats, weight = force_gatherable((loss, stats, weight), loss.device)
        return loss, stats, weight

    def collect_feats(self, **kwargs):
        raise NotImplementedError

    @property
    def layer_cls(self):
        """All layer class that can be warpped by FSDP"""
        return [
            ResidualAttentionBlock,  # Espnet built-in transformer layer.
        ]
