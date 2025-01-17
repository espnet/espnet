#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

logger = logging.getLogger(__name__)

# NOTE: Assumes all speech is same length.


class BeatsPretrainModel(AbsESPnetModel):
    """Beats Pretraining model"""

    @typechecked
    def __init__(
        self,
        encoder: AbsEncoder,
        decoder: nn.Module,
        ignore_id: int = -1,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.ignore_id = ignore_id
        assert (
            encoder.is_pretraining
        ), "Set the encoder to pretraining mode with is_pretraining=True."
        self.encoder = encoder
        self.decoder = decoder
        self.loss_function = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_id, label_smoothing=label_smoothing, reduction="none"
        )
        logger.info(
            f"Initialized BeatsPretrainModel with ignore_id={ignore_id}, "
            f"label_smoothing={label_smoothing}, encoder={encoder}, decoder={decoder}"
        )

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        target: torch.Tensor,
        target_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Encoder + Predictor + Calc loss

        Args:
            speech: (Batch, Length, 1)
            speech_lengths: (Batch, )
            target: (Batch, Length)
            target_lengths: (Batch,)
            kwargs: "utt_id" is among the input.
        """
        assert target_lengths.dim() == 1, target_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == target.shape[0]
            == target_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, target.shape, target_lengths.shape)
        batch_size = speech.shape[0]

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]
        target = target[:, : target_lengths.max()]

        # unmasked_patch_emb (Batch, n_patch*kept_ratio, emb_dim)
        # restore_ids (Batch, n_patch) -- permutation of [0, 1, ..., n_patch-1]
        # kept_mask (Batch, n_patch)
        unmasked_patch_emb, restore_ids, kept_mask = self.encoder(
            speech, speech_lengths
        )

        # target (Batch, n_patch)
        # logits (Batch, n_patch, codebook_size)
        logits = self.decoder(unmasked_patch_emb, target_lengths, restore_ids)

        loss, acc_mask, acc_unmask = self._calc_beats_loss(logits, ~kept_mask, target)

        stats = dict(
            loss=loss.detach(),
            acc_mask=acc_mask,
            acc_unmask=acc_unmask,
            acc=acc_mask,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        self.encoder.is_pretraining = False
        feats, feats_lengths, _ = self.encoder._extract_feats(speech, speech_lengths)
        self.encoder.is_pretraining = True
        return {"feats": feats, "feats_lengths": feats_lengths}

    def _calc_beats_loss(
        self,
        logits: torch.Tensor,
        masked: torch.Tensor,
        target: torch.Tensor,
    ):
        """Compute loss for Beats model.
        Args:
            logits: (Batch, n_patch, codebook_size)
            masked: (Batch, n_patch) -- True for masked, False for unmasked
            target: (Batch, n_patch)
        Returns:
            loss: scalar
            acc_mask: scalar
            acc_unmask: scalar
        """
        logits = logits.transpose(1, 2)  # (Batch, codebook_size, n_patch)
        loss = self.loss_function(logits, target)
        loss = loss * masked  # do not count loss for unmasked patches
        loss = loss.sum()
        # generate a relevant mask from target_lengths
        # relevant_positions = torch.arange(target.shape[1]).unsqueeze(0).repeat(batch_size, 1).to(target.device) < target_lengths.unsqueeze(1)
        # masked = masked & relevant_positions
        # unmasked = unmasked & relevant_positions
        loss = loss / (masked.sum() + 1e-10)  # normalize by number of masked patches
        with torch.no_grad():
            corr_masked = ((logits.argmax(dim=1) == target) * masked).sum().item()
            corr_unmask = ((logits.argmax(dim=1) == target) * (~masked)).sum().item()

            count_unmask = (~masked).sum().item()
            count_masked = (
                masked.sum().item()
            )  # TODO(shikhar): change if input speech lengths are different
        acc_m = corr_masked / (count_masked + 1e-10)
        acc_u = corr_unmask / (count_unmask + 1e-10)
        return loss, acc_m, acc_u


class BeatsTokenizerPretrainModel(AbsESPnetModel):
    pass


def generate_beats_encoder_checkpoint():
    pass
