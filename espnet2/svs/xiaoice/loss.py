# Copyright 2020 Nagoya University (Tomoki Hayashi)
# Copyright 2023 Renmin University of China (Yuning Wu)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""XiaoiceSing2 related loss module for ESPnet2."""

from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.fastspeech.duration_predictor import (  # noqa: H301
    DurationPredictorLoss,
)
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask


class XiaoiceSing2Loss(torch.nn.Module):
    """Loss function module for FastSpeech2."""

    def __init__(self, use_masking: bool = True, use_weighted_masking: bool = False):
        """Initialize feed-forward Transformer loss module.

        Args:
            use_masking (bool): Whether to apply masking for padded part in loss
                calculation.
            use_weighted_masking (bool): Whether to weighted masking in loss
                calculation.

        """
        assert check_argument_types()
        super().__init__()

        assert (use_masking != use_weighted_masking) or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking

        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
        self.l1_criterion = torch.nn.L1Loss(reduction=reduction)
        self.mse_criterion = torch.nn.MSELoss(reduction=reduction)
        self.bce_criterion = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        self.duration_criterion = DurationPredictorLoss(reduction=reduction)

    def forward(
        self,
        after_outs: torch.Tensor,
        before_outs: torch.Tensor,
        d_outs: torch.Tensor,
        p_outs: torch.Tensor,
        v_outs: torch.Tensor,
        ys: torch.Tensor,
        ds: torch.Tensor,
        ps: torch.Tensor,
        vs: torch.Tensor,
        ilens: torch.Tensor,
        olens: torch.Tensor,
        loss_type: str = "L1",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate forward propagation.

        Args:
            after_outs (Tensor): Batch of outputs after postnets (B, T_feats, odim).
            before_outs (Tensor): Batch of outputs before postnets (B, T_feats, odim).
            d_outs (LongTensor): Batch of outputs of duration predictor (B, T_text).
            p_outs (Tensor): Batch of outputs of log_f0 (B, T_text, 1).
            v_outs (Tensor): Batch of outputs of VUV (B, T_text, 1).
            ys (Tensor): Batch of target features (B, T_feats, odim).
            ds (LongTensor): Batch of durations (B, T_text).
            ps (Tensor): Batch of target log_f0 (B, T_text, 1).
            vs (Tensor): Batch of target VUV (B, T_text, 1).
            ilens (LongTensor): Batch of the lengths of each input (B,).
            olens (LongTensor): Batch of the lengths of each target (B,).
            loss_type (str): Mel loss type ("L1" (MAE), "L2" (MSE) or "L1+L2")

        Returns:
            Tensor: Mel loss value.
            Tensor: Duration predictor loss value.
            Tensor: Pitch predictor loss value.
            Tensor: VUV predictor loss value.

        """
        # apply mask to remove padded part
        if self.use_masking:
            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            before_outs = before_outs.masked_select(out_masks)
            if after_outs is not None:
                after_outs = after_outs.masked_select(out_masks)
            ys = ys.masked_select(out_masks)
            duration_masks = make_non_pad_mask(ilens).to(ys.device)
            d_outs = d_outs.masked_select(duration_masks)
            ds = ds.masked_select(duration_masks)
            p_outs = p_outs.masked_select(out_masks)
            v_outs = v_outs.masked_select(out_masks)
            ps = ps.masked_select(out_masks)
            vs = vs.masked_select(out_masks)

        # calculate loss
        if loss_type == "L1":
            mel_loss = self.l1_criterion(before_outs, ys)
            if after_outs is not None:
                mel_loss += self.l1_criterion(after_outs, ys)
        elif loss_type == "L2":
            mel_loss = self.mse_criterion(before_outs, ys)
            if after_outs is not None:
                mel_loss += self.mse_criterion(after_outs, ys)
        elif loss_type == "L1+L2":
            mel_loss = self.l1_criterion(before_outs, ys) + self.mse_criterion(
                before_outs, ys
            )
            if after_outs is not None:
                mel_loss += self.l1_criterion(after_outs, ys) + self.mse_criterion(
                    after_outs, ys
                )
        else:
            raise NotImplementedError("Mel loss support only L1, L2 or L1+L2.")
        duration_loss = self.duration_criterion(d_outs, ds)
        pitch_loss = self.mse_criterion(p_outs, ps)
        vuv_loss = self.bce_criterion(v_outs, vs.float())

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
            mel_loss = mel_loss.mul(out_weights).masked_select(out_masks).sum()
            duration_loss = (
                duration_loss.mul(duration_weights).masked_select(duration_masks).sum()
            )
            pitch_loss = pitch_loss.mul(out_weights).masked_select(out_masks).sum()
            vuv_loss = vuv_loss.mul(out_weights).masked_select(out_masks).sum()

        return mel_loss, duration_loss, pitch_loss, vuv_loss
