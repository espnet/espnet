#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

logger = logging.getLogger(__name__)

# NOTE: Assumes all speech is same length.


class BeatsPretrainModel(AbsESPnetModel):
    """Beats Pretraining model"""

    @typechecked
    def __init__(
        self,
        encoder: AbsEncoder,
        decoder: nn.Module,
        ignore_id: int = -2,
        label_smoothing: float = 0.1,
        sound_input: bool = False,
    ):
        super().__init__()
        self.ignore_id = ignore_id
        assert (
            encoder.is_pretraining
        ), "Set the encoder to pretraining mode with is_pretraining=True."
        self.encoder = encoder
        self.decoder = decoder
        self.sound_input = sound_input
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
            speech: (Batch, Length, Dim). Either raw speech or features.
                    If raw speech, then should be single channel ie Dim=1.
            speech_lengths: (Batch, )
            target: (Batch, Length)
            target_lengths: (Batch,)
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
        # patch_len (Batch,)
        unmasked_patch_emb, patch_len, restore_ids, kept_mask = self.encoder(
            speech, speech_lengths, is_sound_input=self.sound_input
        )
        target = target[:, : patch_len.max()]

        # target (Batch, n_patch)
        # logits (Batch, n_patch, codebook_size)
        logits = self.decoder(unmasked_patch_emb, patch_len, restore_ids)

        loss, stats = self._calc_beats_loss(
            logits, target - 1, ~kept_mask, patch_len
        )  # target - 1 because of unk token at 0th position

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        target: torch.Tensor,
        target_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        old_is_pretraining = self.encoder.is_pretraining
        self.encoder.is_pretraining = False
        # for data-parallel
        speech = speech[:, : speech_lengths.max()]
        if self.sound_input:
            feats, feats_lengths = self.encoder.preprocess(speech)
        else:
            feats, feats_lengths = speech, speech_lengths
        self.encoder.is_pretraining = old_is_pretraining
        return {"feats": feats, "feats_lengths": feats_lengths}

    def _calc_beats_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        masked: torch.Tensor,
        speech_lengths: torch.Tensor,
    ):
        """Compute loss for Beats model.
        Args:
            logits: (Batch, n_patch, codebook_size)
            target: (Batch, n_patch)
            masked: (Batch, n_patch) -- True for masked, False for unmasked
            speech_lengths: (Batch, )
        Returns:
            loss: scalar
            acc_mask: scalar
            acc_unmask: scalar
        """
        logits = logits.transpose(1, 2)  # (Batch, codebook_size, n_patch)
        loss = self.loss_function(logits, target)
        loss = loss * masked  # do not count loss for unmasked patches
        loss = loss.sum()

        padding_mask = make_pad_mask(speech_lengths, traceable=False).to(loss.device)
        unmasked = ~masked & ~padding_mask  # not masked and not padded
        masked = masked & ~padding_mask  # masked and not padded, no-op

        weight = masked.sum() + 1e-10
        loss = loss / weight  # normalize by number of masked patches
        preds = logits.argmax(dim=1)
        with torch.no_grad():
            corr_masked = ((preds == target) * masked).sum()
            corr_unmask = ((preds == target) * unmasked).sum()

            count_unmask = unmasked.sum()
            count_masked = masked.sum()
            acc_m = corr_masked / (count_masked + 1e-10)
            acc_u = corr_unmask / (count_unmask + 1e-10)
            n_uniq_pred = torch.unique(preds, return_counts=False).numel()
            vocab_cov_pred = n_uniq_pred / logits.shape[1]
            n_uniq_pred_masked = torch.unique(
                preds[masked], return_counts=False
            ).numel()
            n_uniq_pred_unmask = torch.unique(
                preds[unmasked], return_counts=False
            ).numel()

            n_uniq_tgt_masked = torch.unique(
                target[masked], return_counts=False
            ).numel()
            n_uniq_tgt = torch.unique(target, return_counts=False).numel()
            vocab_cov_tgt = n_uniq_tgt / logits.shape[1]

            probs = torch.nn.functional.softmax(logits, dim=1)
            entropy = -(probs * probs.log()).sum(dim=1).mean()

        # Note(shikhar): Some of these are redundant
        stats_dict = dict(
            loss=loss.detach(),
            acc=acc_m,
            # Masked instance metrics
            acc_mask=acc_m,
            count_masked=count_masked * 1.0,
            n_uniq_tgt_msk=n_uniq_tgt_masked * 1.0,
            n_uniq_pred_msk=n_uniq_pred_masked * 1.0,
            # Unmasked instance metrics
            acc_unmask=acc_u,
            count_unmask=count_unmask * 1.0,
            n_uniq_pred_unmsk=n_uniq_pred_unmask * 1.0,
            n_uniq_pred=n_uniq_pred * 1.0,
            n_uniq_tgt=n_uniq_tgt * 1.0,
            # Vocab coverage metrics
            vocab_cov_tgt=vocab_cov_tgt,
            vocab_cov_pred=vocab_cov_pred,
            entropy=entropy,
        )
        return loss, stats_dict


class BeatsTokenizerPretrainModel(AbsESPnetModel):

    @typechecked
    def __init__(
        self,
        encoder: AbsEncoder,
        decoder: nn.Module,
        teacher: AbsEncoder,
    ):
        super().__init__()
        assert (
            encoder.is_tokenizer_pretraining
        ), "Set the encoder to pretraining mode with is_tokenizer_pretraining=True."
        self.encoder = encoder  # BEATs tokenizer model
        self.decoder = decoder  # BEATs tokenizer predictor
        self.teacher = teacher  # BEATs audio encoder
        self.teacher.eval()
        assert (
            not self.teacher.is_pretraining
        ), "Teacher should not be in pretraining mode."
        assert (
            not self.encoder.is_pretraining
        ), "Tokenizer should not be in encoder pretraining mode."
        assert (
            self.encoder.is_tokenizer_pretraining
        ), "Tokenizer should be in tokenizer pretraining mode."
        logger.info(
            f"Initialized BeatsPretrainModel with"
            f"encoder={encoder}, decoder={decoder}, teacher={teacher}"
        )

    @torch.no_grad()
    def _extract_teacher_targets(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ):
        audio_representation, output_lens, _ = self.teacher(speech, speech_lengths)
        return audio_representation, output_lens

    def collect_feats(self, speech: torch.Tensor, speech_lengths: torch.Tensor):
        # for data-parallel
        speech = speech[:, : speech_lengths.max()]
        _, _, feats, feats_lengths = self.encoder.encode(speech)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def _calc_beats_tokenizer_loss(
        self, output: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor
    ):
        cos_sim = F.cosine_similarity(target, output, dim=-1)
        pad_mask = make_pad_mask(lengths, traceable=False).to(
            cos_sim.device
        )  # can optimize
        cos_sim[pad_mask] = 0
        cos_loss = 1 - (cos_sim.sum() / lengths.sum())
        return cos_loss

    def forward(self, speech: torch.Tensor, speech_lengths: torch.Tensor):
        assert speech.shape[0] == speech_lengths.shape[0], (
            speech.shape,
            speech_lengths.shape,
        )
        batch_size = speech.shape[0]
        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        targets, target_lengths = self._extract_teacher_targets(speech, speech_lengths)
        _, embed_loss, quantize_feature, quantize_feats_len = self.encoder.encode(
            speech, speech_lengths
        )
        assert (quantize_feats_len == target_lengths).all(), "Mismatch in lengths"
        tokenizer_features = self.decoder(quantize_feature, quantize_feats_len)
        sim_loss = self._calc_beats_tokenizer_loss(
            tokenizer_features, targets, target_lengths
        )
        loss = embed_loss + sim_loss
        stats = dict(
            loss=loss.detach(),
            embed_loss=embed_loss.detach(),
            similarity_loss=sim_loss.detach(),
            # codebook_coverage=self.encoder.quantize.cluster_size,
        )
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight
