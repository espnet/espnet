# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""FastSpeech related modules."""

import logging

import torch
import torch.nn.functional as F

from espnet.asr.asr_utils import get_model_conf, torch_load
from espnet.nets.pytorch_backend.fastspeech.duration_calculator import (  # noqa: H301
    DurationCalculator,
)
from espnet.nets.pytorch_backend.fastspeech.duration_predictor import (  # noqa: H301
    DurationPredictor,
    DurationPredictorLoss,
)
from espnet.nets.pytorch_backend.fastspeech.length_regulator import LengthRegulator
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask, make_pad_mask
from espnet.nets.pytorch_backend.tacotron2.decoder import Postnet
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,
    ScaledPositionalEncoding,
)
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.tts_interface import TTSInterface
from espnet.utils.cli_utils import strtobool
from espnet.utils.fill_missing_args import fill_missing_args


class DiscreteLoss(torch.nn.Module):
    """Loss function module for feed-forward Transformer."""

    def __init__(
        self, use_masking=True, use_weighted_masking=False, predict_pitch=False
    ):
        """Initialize feed-forward Transformer loss module.

        Args:
            use_masking (bool):
                Whether to apply masking for padded part in loss calculation.
            use_weighted_masking (bool):
                Whether to weighted masking in loss calculation.
            predict_pitch (bool):
                Whether to predict pitch and calculate pitch loss.
        """
        super(DiscreteLoss, self).__init__()
        assert (use_masking != use_weighted_masking) or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking
        self.predict_pitch = predict_pitch

        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
        # self.l1_criterion = torch.nn.L1Loss(reduction=reduction)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction=reduction)
        self.duration_criterion = DurationPredictorLoss(reduction=reduction)
        if self.predict_pitch:
            self.pitch_criterion = torch.nn.MSELoss(reduction=reduction)

    def forward(
        self,
        after_outs: torch.Tensor,
        before_outs: torch.Tensor,
        d_outs: torch.Tensor,
        ys: torch.Tensor,
        ds: torch.Tensor,
        ilens: torch.Tensor,
        olens: torch.Tensor,
        p_outs: torch.Tensor = None,
        ps: torch.Tensor = None,
    ):
        """Calculate forward propagation.

        Args:
            after_outs (Tensor): Batch of outputs after postnets (B, Lmax, odim).
            before_outs (Tensor): Batch of outputs before postnets (B, Lmax, odim).
            d_outs (Tensor): Batch of outputs of duration predictor (B, Tmax).
            ys (Tensor): Batch of target features (B, Lmax, odim).
            ds (Tensor): Batch of durations (B, Tmax).
            ilens (LongTensor): Batch of the lengths of each input (B,).
            olens (LongTensor): Batch of the lengths of each target (B,).
            p_outs (Tensor): Batch of outputs of log_f0 (B, T_text, 1).
            ps (Tensor): Batch of target log_f0 (B, T_text, 1).

        Returns:
            Tensor: L1 loss value.
            Tensor: Duration predictor loss value.
            Tensor: Pitch loss value.

        """
        # apply mask to remove padded part
        dim = after_outs.shape[-1]
        if self.use_masking:
            duration_masks = make_non_pad_mask(ilens).to(ys.device)
            d_outs = d_outs.masked_select(duration_masks)
            ds = ds.masked_select(duration_masks)
            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            before_outs = before_outs.masked_select(out_masks).reshape(-1, dim)
            after_outs = (
                after_outs.masked_select(out_masks) if after_outs is not None else None
            ).reshape(-1, dim)
            if self.predict_pitch:
                p_outs = p_outs.masked_select(out_masks)
                ps = ps.masked_select(out_masks)
            out_masks = make_non_pad_mask(olens).to(ys.device)
            ys = ys.masked_select(out_masks)

        # calculate loss
        CE_loss = self.cross_entropy(before_outs, ys)
        if after_outs is not None:
            CE_loss += self.cross_entropy(after_outs, ys)
        duration_loss = self.duration_criterion(d_outs, ds)
        if self.predict_pitch:
            pitch_loss = self.pitch_criterion(p_outs, ps)

        # make weighted mask and apply it
        if self.use_weighted_masking:
            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            out_weights = out_masks.float() / out_masks.sum(dim=1, keepdim=True).float()
            out_weights /= ys.size(0) * ys.size(2)
            duration_masks = make_non_pad_mask(ilens).to(ys.device)
            duration_weights = (
                duration_masks.float() / duration_masks.sum(dim=1, keepdim=True).float()
            )
            duration_weights /= ds.size(0)

            # apply weight
            l1_loss = l1_loss.mul(out_weights).masked_select(out_masks).sum()
            duration_loss = (
                duration_loss.mul(duration_weights).masked_select(duration_masks).sum()
            )
            if self.predict_pitch:
                pitch_loss = pitch_loss.mul(out_weights).masked_select(out_masks).sum()

        if self.predict_pitch:
            return CE_loss, duration_loss, pitch_loss
        else:
            return CE_loss, duration_loss
