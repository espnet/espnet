#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from typing import Dict, Optional, Tuple, List

import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet2.speechlm.core_lm.abs_core_lm import AbsCoreLM
from espnet2.speechlm.predictor.abs_predictor import AbsPredictor
from espnet2.speechlm.postprocessor.abs_postprocessor import AbsPostProcessor
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.speechlm.net_utils import length_mask


class ESPnetSpeechLMModel(AbsESPnetModel):
    def __init__(
        self,
        nq: int,
        token_list: List,
        corelm: AbsCoreLM,
        predictor: AbsPredictor,
        postprocessor: Optional[AbsPostProcessor],
        share_emb: bool = True,
        extract_feats_in_collect_stats: bool = False,
    ):
        assert check_argument_types()
        super().__init__()

        # modules
        self.corelm = corelm
        self.predictor = predictor
        self.post_processor = postprocessor
        self.emb = torch.nn.Embedding(len(token_list), corelm.model_dim)
        if share_emb:
            self.emb.data = predictor.get_lookup_table()

        # special tokens
        self.token_list = token_list

        # configurations
        self.nq = nq
        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    def forward(
        self,
        dec_seq: torch.Tensor,
        dec_seq_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:

        ### (1) embeddings. All representations in the shape [B, T, nq, D]
        # (1.1) decoder
        dec_seq = dec_seq[:, :max(dec_seq_lengths)]
        dec_emb = self.emb(dec_seq)
        dec_emb, dec_seq_lengths = dec_emb[:, :-1], dec_seq_lengths - 1 # shift by one

        # (1.2) encoder
        enc_seq = kwargs.get("enc_seq", None)
        enc_seq_lengths = kwargs.get("enc_seq_lengths", None)
        if enc_seq is not None and enc_seq_lengths is not None:
            enc_seq = enc_seq[:, :max(enc_seq_lengths)]
            enc_emb = self.emb(enc_seq)
        else:
            enc_emb = None
        
        # (1.3) target
        target, target_lengths = dec_seq[:, 1:], dec_seq_lengths
        target_emb = self.emb(target)
        
        ### (2) CoreLM. Prediction in shape [B, T, D] or [B, T, *, D]
        pred, pred_lengths, others = self.corelm(
            dec_emb,
            dec_seq_lengths,
            enc_emb,
            enc_seq_lengths,
        )

        ### (3) predictor. Logits and target in shape [B, T, nq, V] and [B, T, nq]
        logits, logits_lengths, others = self.predictor(
            pred,
            pred_lengths,
            target_emb,
            target_lengths,
            others,
        )
        target, target_lengths, others = self.predictor.organize_target(
            target, target_lengths, others
        )

        ###(4) Loss computing
        loss, stats, weight = self.compute_loss(
            logits,
            logits_lengths,
            target,
            target_lengths,
            others,
        )

        loss, stats, weight = force_gatherable((loss, stats, weight), loss.device)
        return loss, stats, weight

    def compute_loss(
        self,
        logits: torch.Tensor,
        logits_lengths: torch.Tensor,
        target_sequence: torch.Tensor,
        target_sequence_lengths: torch.Tensor,
        others: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert torch.all(torch.eq(logits_lengths, target_sequence_lengths))
        assert logits.size()[:-1] == target_sequence.size()
        assert logits_lengths.max() == logits.size(1)

        elem_loss = torch.nn.functional.cross_entropy(
            logits.permute(0, 3, 1, 2), target_sequence, reduction="none"
        )

        mask = length_mask(logits_lengths).to(elem_loss.dtype).unsqueeze(-1)
        elem_loss = elem_loss * mask
        loss = elem_loss.sum() / mask.sum() / self.nq

        pred = logits.argmax(dim=-1)
        acc = torch.eq(pred, target_sequence).to(elem_loss.dtype) * mask
        
        stats = {}
        for nq_idx in range(target_sequence.size(2)):
            stats.update({
                f"acc_layer{nq_idx}": acc[:, :, nq_idx].sum() / mask.sum()
            })
        
        acc = acc.sum() / mask.sum() / self.nq
        stats.update({"loss": loss.clone().detach(), "acc": acc})
        weight = mask.sum()

        return loss, stats, weight

    def collect_feats(self, **kwargs):
        raise NotImplementedError