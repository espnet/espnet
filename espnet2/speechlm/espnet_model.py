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
        self.emb = torch.nn.Embedding(len(token_list), corelm.model_dim())
        if share_emb:
            self.emb.data = predictor.get_lookup_table()

        # special tokens
        self.token_list = token_list

        # configurations
        self.nq = nq  # N_q
        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    def forward(
        self,
        decoder_sequence: torch.Tensor,
        decoder_sequence_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:

        encoder_sequence = kwargs.get("encoder_sequence", None)
        encoder_sequence_lengths = kwargs.get("encoder_sequence_lengths", None)

        # (1) embeddings
        # encoder / decoder shape: [B, T, nq] -> [B, T - 1, D]
        decoder_input_emb, decoder_input_lengths = self.embed_forward(
            decoder_sequence[:, :-1], decoder_sequence_lengths - 1
        )
        decoder_input_emb = decoder_input_emb.sum(dim=2)
        if encoder_sequence is not None and encoder_sequence_lengths is not None:
            encoder_input_emb, encoder_input_lengths = self.embed_forward(
                encoder_sequence, encoder_sequence_lengths
            )
            encoder_input_emb = encoder_input_emb.sum(dim=2)
        else:
            encoder_input_emb, encoder_input_lengths = None, None
        
        # (2) corelm forward
        # pred_emb: [B, T, D]
        pred_emb, pred_lengths = self.corelm_forward(
            decoder_input_emb,
            decoder_input_lengths,
            encoder_input_emb,
            encoder_input_lengths,
        )

        # (3) predictor forward
        # pred_emb: [B, T-1, D] -> [B, T-1, nq, D]
        target, target_lengths = decoder_sequence[:, 1:], decoder_sequence_lengths - 1
        pred_emb, pred_lengths, target, target_lengths = self.predictor_forward(
            pred_emb,
            pred_lengths,
            target,
            target_lengths,  # was down-sampled in embed stage
        )

        # (4) Loss computing
        loss, stats, weight = self.compute_loss(
            pred_emb,
            pred_lengths,
            target_sequence=target,
            target_sequence_lengths=target_lengths,
        )

        loss, stats, weight = force_gatherable((loss, stats, weight), loss.device)
        return loss, stats, weight

    def embed_forward(
        self,
        sequence: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        sequence = sequence[:, :lengths.max()]
        embeddings = self.emb(sequence)

        return embeddings, lengths

    def corelm_forward(
        self,
        decoder_sequence_emb: torch.Tensor,
        decoder_sequence_lengths: torch.Tensor,
        encoder_sequence_emb: torch.Tensor,
        encoder_sequence_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.corelm(
            decoder_sequence_emb,
            decoder_sequence_lengths,
            encoder_sequence_emb,
            encoder_sequence_lengths,
        )

    def predictor_forward(
        self,
        pred_emb: torch.Tensor,
        pred_lengths: torch.Tensor,
        target: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        pred_emb, pred_lengths = self.predictor(
            pred_emb,
            pred_lengths,
            self.emb(target),
            target_lengths,
        )
        target, target_lengths = self.predictor.organize_target(target, target_lengths)

        return pred_emb, pred_lengths, target, target_lengths

    def compute_loss(
        self,
        pred_emb: torch.Tensor,
        pred_emb_lengths: torch.Tensor,
        target_sequence: torch.Tensor,
        target_sequence_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert torch.all(torch.eq(pred_emb_lengths, target_sequence_lengths))
        assert pred_emb.size()[:-1] == target_sequence.size()
        assert pred_emb_lengths.max() == pred_emb.size(1)

        elem_loss = torch.nn.functional.cross_entropy(
            pred_emb.permute(0, 3, 1, 2), target_sequence, reduction="none"
        )

        mask = length_mask(pred_emb_lengths).to(elem_loss.dtype).unsqueeze(-1)
        elem_loss = elem_loss * mask
        loss = elem_loss.sum() / mask.sum() / self.nq

        pred = pred_emb.argmax(dim=-1)
        acc = torch.eq(pred, target_sequence).to(elem_loss.dtype) * mask
        acc = acc.sum() / mask.sum() / self.nq

        stats = {"loss": loss.clone().detach(), "acc": acc}
        weight = mask.sum()

        return loss, stats, weight

    def collect_feats(self, **kwargs):
        raise NotImplementedError