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
        ignore_id: int = -1,
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
        unmasked_patch_emb, restore_ids, kept_mask = self.encoder(
            speech, speech_lengths, is_sound_input=self.sound_input
        )

        # target (Batch, n_patch)
        # logits (Batch, n_patch, codebook_size)
        logits = self.decoder(unmasked_patch_emb, target_lengths, restore_ids)

        loss, acc_mask, acc_unmask, vocab_cov = self._calc_beats_loss(
            logits, ~kept_mask, target - 1
        )  # target - 1 because of unk token at 0th position

        stats = dict(
            loss=loss.detach(),
            acc_mask=acc_mask,
            acc_unmask=acc_unmask,
            acc=acc_mask,
            vocab_cov=vocab_cov,
        )

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

            # TODO(shikhar): change if input speech lengths are different
            count_unmask = (~masked).sum().item()
            count_masked = masked.sum().item()
        acc_m = corr_masked / (count_masked + 1e-10)
        acc_u = corr_unmask / (count_unmask + 1e-10)

        n_uniq = target.unique().shape[0]
        vocab_cov = n_uniq / logits.shape[1]
        return loss, acc_m, acc_u, vocab_cov


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
        pad_mask = make_pad_mask(lengths, traceable=False).to(cos_sim.device) # can optimize
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


def generate_beats_encoder_checkpoint(espnet_model_checkpoint_path: str, output_path: str):
    """Generate a checkpoint for Encoder from Pretraining model checkpoint."""
    print('here!!')
    espnet_state_dict = torch.load(espnet_model_checkpoint_path, map_location="cpu")
    new_state_dict = {'model': {}, 'cfg': {}}
    for key, value in espnet_state_dict.items():
        if key.startswith('encoder.'):
            new_state_dict['model'][key[len('encoder.'):]] = value
    torch.save(new_state_dict, output_path)


def get_cmdline_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a checkpoint for Encoder from Pretraining model checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--espnet_model_checkpoint_path",
        type=str,
        required=True,
        help="Path to ESPnet model checkpoint",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the new checkpoint",
    )
    return parser

def main(cmd=None):
    import sys
    from espnet.utils.cli_utils import get_commandline_args
    print(get_commandline_args(), file=sys.stderr)
    parser = get_cmdline_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    logger.info(f"Kwargs: {kwargs}")
    generate_beats_encoder_checkpoint(**kwargs)

if __name__ == "__main__":
    main()