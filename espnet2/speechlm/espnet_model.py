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
        loss_mask: torch.Tensor,
        conti_feats: list,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:

        logits, targets, loss_mask = self.corelm(
            dec_seq,
            loss_mask,
            conti_feats,
        )

        loss, _, stats, weight = self.criterion(
            logits,
            targets,
            loss_mask,
        )
        loss, stats, weight = force_gatherable((loss, stats, weight), loss.device)
        return loss, stats, weight

    def collect_feats(self, **kwargs):
        raise NotImplementedError
