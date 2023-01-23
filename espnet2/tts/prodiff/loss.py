# Copyright 2022 Hitachi LTD. (Nelson Yalta)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""ProDiff related loss module for ESPnet2."""

from math import exp
from typing import Tuple

import torch
from torch.nn import functional as F
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.fastspeech.duration_predictor import (  # noqa: H301
    DurationPredictorLoss,
)
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    """Gaussian Noise.

    Args:
        window_size (int): Window size.
        sigma (float): Noise sigma.

    Returns:
        torch.Tensor: Noise.

    """
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


class SSimLoss(torch.nn.Module):
    """SSimLoss.

    This is an implementation of structural similarity (SSIM) loss.
    This code is modified from https://github.com/Po-Hsun-Su/pytorch-ssim.

    """

    def __init__(
        self,
        bias: float = 6.0,
        window_size: int = 11,
        channels: int = 1,
        reduction: str = "none",
    ):
        """Initialization.

        Args:
            bias (float, optional): value of the bias. Defaults to 6.0.
            window_size (int, optional): Window size. Defaults to 11.
            channels (int, optional): Number of channels. Defaults to 1.
            reduction (str, optional): Type of reduction during the loss
                calculation. Defaults to "none".

        """
        super().__init__()
        self.bias = bias
        self.win_len = window_size
        self.channels = channels
        self.average = False
        if reduction == "mean":
            self.average = True

        win1d = gaussian(window_size, 1.5).unsqueeze(1)
        win2d = win1d.mm(win1d.t()).float().unsqueeze(0).unsqueeze(0)
        self.window = torch.Tensor(
            win2d.expand(channels, 1, window_size, window_size).contiguous()
        )

    def forward(self, outputs: torch.Tensor, target: torch.Tensor):
        """Calculate forward propagation.

        Args:
            outputs (torch.Tensor): Batch of output sequences generated
                by the model (batch, time, mels).
            target (torch.Tensor): Batch of sequences with true
                states (batch, time, mels).

        Returns:
            Tensor: Loss scalar value.

        """
        with torch.no_grad():
            dim = target.size(-1)
            mask = target.abs().sum(-1, keepdim=True).ne(0).float().repeat(1, 1, dim)
        outputs = outputs.unsqueeze(1) + self.bias
        target = target.unsqueeze(1) + self.bias
        loss = 1 - self.ssim(outputs, target)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def ssim(self, tensor1: torch.Tensor, tensor2: torch.Tensor):
        """Calculate SSIM loss.

        Args:
            tensor1 (torch.Tensor): Generated output.
            tensor2 (torch.Tensor): Groundtruth output.

        Returns:
            Tensor: Loss scalar value.

        """
        window = self.window.to(tensor1.device)
        mu1 = F.conv2d(tensor1, window, padding=self.win_len // 2, groups=self.channels)
        mu2 = F.conv2d(tensor2, window, padding=self.win_len // 2, groups=self.channels)
        mu_corr = mu1 * mu2

        mu1 = mu1.pow(2)
        mu2 = mu2.pow(2)

        sigma1 = (
            F.conv2d(
                tensor1 * tensor1,
                window,
                padding=self.win_len // 2,
                groups=self.channels,
            )
            - mu1
        )

        sigma2 = (
            F.conv2d(
                tensor2 * tensor2,
                window,
                padding=self.win_len // 2,
                groups=self.channels,
            )
            - mu2
        )

        sigma_corr = (
            F.conv2d(
                tensor1 * tensor2,
                window,
                padding=self.win_len // 2,
                groups=self.channels,
            )
            - mu_corr
        )

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu_corr + C1) * (2 * sigma_corr + C2)) / (
            (mu1 + mu2 + C1) * (sigma1 + sigma2 + C2)
        )
        if self.average:
            return ssim_map.mean()
        return ssim_map.mean(1)


class ProDiffLoss(torch.nn.Module):
    """Loss function module for ProDiffLoss."""

    def __init__(
        self,
        use_masking: bool = True,
        use_weighted_masking: bool = False,
    ):
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
        self.duration_criterion = DurationPredictorLoss(reduction=reduction)
        self.ssim_criterion = SSimLoss(reduction="none")

    def forward(
        self,
        after_outs: torch.Tensor,
        before_outs: torch.Tensor,
        d_outs: torch.Tensor,
        p_outs: torch.Tensor,
        e_outs: torch.Tensor,
        ys: torch.Tensor,
        ds: torch.Tensor,
        ps: torch.Tensor,
        es: torch.Tensor,
        ilens: torch.Tensor,
        olens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate forward propagation.

        Args:
            after_outs (Tensor): Batch of outputs after postnets (B, T_feats, odim).
            before_outs (Tensor): Batch of outputs before postnets (B, T_feats, odim).
            d_outs (LongTensor): Batch of outputs of duration predictor (B, T_text).
            p_outs (Tensor): Batch of outputs of pitch predictor (B, T_text, 1).
            e_outs (Tensor): Batch of outputs of energy predictor (B, T_text, 1).
            ys (Tensor): Batch of target features (B, T_feats, odim).
            ds (LongTensor): Batch of durations (B, T_text).
            ps (Tensor): Batch of target token-averaged pitch (B, T_text, 1).
            es (Tensor): Batch of target token-averaged energy (B, T_text, 1).
            ilens (LongTensor): Batch of the lengths of each input (B,).
            olens (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: L1 loss value.
            Tensor: Duration predictor loss value.
            Tensor: Pitch predictor loss value.
            Tensor: Energy predictor loss value.

        """
        # First SSIM before masks
        ssim_loss = self.ssim_criterion(before_outs, ys)

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
            pitch_masks = make_non_pad_mask(ilens).unsqueeze(-1).to(ys.device)
            p_outs = p_outs.masked_select(pitch_masks)
            e_outs = e_outs.masked_select(pitch_masks)
            ps = ps.masked_select(pitch_masks)
            es = es.masked_select(pitch_masks)

        # calculate loss
        l1_loss = self.l1_criterion(before_outs, ys)
        if after_outs is not None:
            l1_loss += self.l1_criterion(after_outs, ys)

        duration_loss = self.duration_criterion(d_outs, ds)
        pitch_loss = self.mse_criterion(p_outs, ps)
        energy_loss = self.mse_criterion(e_outs, es)

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
            pitch_masks = duration_masks.unsqueeze(-1)
            pitch_weights = duration_weights.unsqueeze(-1)
            pitch_loss = pitch_loss.mul(pitch_weights).masked_select(pitch_masks).sum()
            energy_loss = (
                energy_loss.mul(pitch_weights).masked_select(pitch_masks).sum()
            )

        return l1_loss, ssim_loss, duration_loss, pitch_loss, energy_loss
