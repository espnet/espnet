# Copyright 2019 Tomoki Hayashi
# Copyright 2021 Renmin University of China (Shuai Guo)
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


class DiffLoss(torch.nn.Module):
    """Loss function module for feed-forward Transformer."""

    def __init__(self, use_masking=True, use_weighted_masking=False):
        """Initialize feed-forward Transformer loss module.
        Args:
            use_masking (bool):
                Whether to apply masking for padded part in loss calculation.
            use_weighted_masking (bool):
                Whether to weighted masking in loss calculation.
        """
        super(DiffLoss, self).__init__()
        assert (use_masking != use_weighted_masking) or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking

        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
        self.l1_criterion = torch.nn.L1Loss(reduction=reduction)
        self.l2_criterion = torch.nn.MSELoss(reduction=reduction)
        self.duration_criterion = DurationPredictorLoss(reduction=reduction)

    def forward(self, noise, predict_noise, noise_mask, d_outs, ds, ilens, loss_type):
        """Calculate forward propagation.
        Args:
            noise (Tensor): Batch of Guass noise (B, 1, T_feats, odim).
            predict_noise (Tensor): Batch of predicted noise (B, 1, T_feats, odim).
            noise_mask (Tensor): Batch of mask of noise (B, T_feats, odim)
            d_outs (Tensor): Batch of outputs of duration predictor (B, Tmax).
            ds (Tensor): Batch of durations (B, Tmax).
            ilens (LongTensor): Batch of the lengths of each input (B,).
            loss_type (Str): loss type in 'l1' or 'l2'
        Returns:
            Tensor: L1/L2 loss value.
            Tensor: Duration predictor loss value.
        """
        # apply mask to remove padded part
        if self.use_masking:
            duration_masks = make_non_pad_mask(ilens).to(ds.device)
            d_outs = d_outs.masked_select(duration_masks)
            ds = ds.masked_select(duration_masks)
            if noise_mask is not None:
                noise = noise.masked_select(noise_mask.unsqueeze(1))
                predict_noise = predict_noise.masked_select(noise_mask.unsqueeze(1))

        # calculate loss
        l1_loss = self.l1_criterion(noise, predict_noise)
        l2_loss = self.l2_criterion(noise, predict_noise)
        duration_loss = self.duration_criterion(d_outs, ds)

        # make weighted mask and apply it
        if self.use_weighted_masking:
            duration_masks = make_non_pad_mask(ilens).to(ds.device)
            duration_weights = (
                duration_masks.float() / duration_masks.sum(dim=1, keepdim=True).float()
            )
            duration_weights /= ds.size(0)

            # apply weight
            duration_loss = (
                duration_loss.mul(duration_weights).masked_select(duration_masks).sum()
            )

        if loss_type == "L1":
            l2_loss = torch.tensor(0.0).to(ds.device)
        else:
            l1_loss = torch.tensor(0.0).to(ds.device)

        return l1_loss, l2_loss, duration_loss
