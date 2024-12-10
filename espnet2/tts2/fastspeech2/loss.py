# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Fastspeech2 related loss module for ESPnet2. Speech Target are discrete units """

from typing import Tuple

import torch
from typeguard import typechecked

from espnet.nets.pytorch_backend.fastspeech.duration_predictor import (  # noqa: H301
    DurationPredictorLoss,
)
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask


class FastSpeech2LossDiscrete(torch.nn.Module):
    """
    Loss function module for FastSpeech2, designed for calculating various
    loss components used in training the FastSpeech2 model. This module
    computes the cross-entropy loss for discrete features, as well as losses
    for duration, pitch, and energy predictors.

    Attributes:
        use_masking (bool): Indicates whether to apply masking for padded parts
            in loss calculations.
        use_weighted_masking (bool): Indicates whether to use weighted masking
            in loss calculations.
        ce_criterion (torch.nn.CrossEntropyLoss): Cross-entropy loss criterion.
        mse_criterion (torch.nn.MSELoss): Mean squared error loss criterion.
        duration_criterion (DurationPredictorLoss): Duration prediction loss criterion.

    Args:
        use_masking (bool): Whether to apply masking for padded part in loss
            calculation. Default is True.
        use_weighted_masking (bool): Whether to apply weighted masking in loss
            calculation. Default is False.
        ignore_id (int): Index to ignore in the loss calculation. Default is -1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            CrossEntropy loss value, Duration predictor loss value,
            Pitch predictor loss value, Energy predictor loss value.

    Examples:
        >>> loss_module = FastSpeech2LossDiscrete()
        >>> ce_loss, duration_loss, pitch_loss, energy_loss = loss_module(
        ...     after_outs, before_outs, d_outs, p_outs, e_outs, ys, ds, ps, es, ilens, olens
        ... )

    Note:
        This class is part of the ESPnet2 toolkit and is specifically designed
        for FastSpeech2, which utilizes discrete speech targets.

    Raises:
        AssertionError: If both `use_masking` and `use_weighted_masking`
            are set to True.
    """

    @typechecked
    def __init__(
        self,
        use_masking: bool = True,
        use_weighted_masking: bool = False,
        ignore_id: int = -1,
    ):
        """Initialize feed-forward Transformer loss module.

        Args:
            use_masking (bool): Whether to apply masking for padded part in loss
                calculation.
            use_weighted_masking (bool): Whether to weighted masking in loss
                calculation.

        """
        super().__init__()

        assert (use_masking != use_weighted_masking) or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking

        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
        self.ce_criterion = torch.nn.CrossEntropyLoss(
            reduction=reduction, ignore_index=ignore_id
        )
        self.mse_criterion = torch.nn.MSELoss(reduction=reduction)
        self.duration_criterion = DurationPredictorLoss(reduction=reduction)

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
        """
                Calculate forward propagation for the FastSpeech2 model, computing the loss
        values for cross-entropy, duration, pitch, and energy predictions.

        Args:
            after_outs (torch.Tensor): Batch of outputs after postnets
                (B, T_feats, odim).
            before_outs (torch.Tensor): Batch of outputs before postnets
                (B, T_feats, odim).
            d_outs (torch.LongTensor): Batch of outputs of duration predictor
                (B, T_text).
            p_outs (torch.Tensor): Batch of outputs of pitch predictor
                (B, T_text, 1).
            e_outs (torch.Tensor): Batch of outputs of energy predictor
                (B, T_text, 1).
            ys (torch.Tensor): Batch of target features in discrete space
                (B, T_feats).
            ds (torch.LongTensor): Batch of durations (B, T_text).
            ps (torch.Tensor): Batch of target token-averaged pitch
                (B, T_text, 1).
            es (torch.Tensor): Batch of target token-averaged energy
                (B, T_text, 1).
            ilens (torch.LongTensor): Batch of the lengths of each input (B,).
            olens (torch.LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                A tuple containing the following loss values:
                - CrossEntropy loss value.
                - Duration predictor loss value.
                - Pitch predictor loss value.
                - Energy predictor loss value.

        Examples:
            >>> loss_function = FastSpeech2LossDiscrete()
            >>> ce_loss, duration_loss, pitch_loss, energy_loss = loss_function(
            ...     after_outs, before_outs, d_outs, p_outs, e_outs, ys, ds, ps, es, ilens, olens
            ... )

        Note:
            The method applies masking to handle padded parts in the loss calculation
            if `use_masking` is set to True. If `use_weighted_masking` is also True,
            weighted masking will be applied for loss computation.
        """
        if len(before_outs.size()) == 3:
            batch_size, max_len, vocab_size = before_outs.size()
        else:
            batch_size, max_len, discrete_token_layers, vocab_size = before_outs.size()
        # apply mask to remove padded part
        if self.use_masking:
            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            if len(before_outs.size()) > 3:
                out_masks = out_masks.unsqueeze(-1).repeat(
                    1, 1, discrete_token_layers, 1
                )
            before_outs = before_outs.masked_select(out_masks).view(-1, vocab_size)
            if after_outs is not None:
                after_outs = after_outs.masked_select(out_masks).view(-1, vocab_size)
            ys = ys.masked_select(out_masks.squeeze(-1))
            duration_masks = make_non_pad_mask(ilens).to(ys.device)
            d_outs = d_outs.masked_select(duration_masks)
            ds = ds.masked_select(duration_masks)
            pitch_masks = make_non_pad_mask(ilens).unsqueeze(-1).to(ys.device)
            p_outs = p_outs.masked_select(pitch_masks)
            e_outs = e_outs.masked_select(pitch_masks)
            ps = ps.masked_select(pitch_masks)
            es = es.masked_select(pitch_masks)
        else:
            before_outs = before_outs.view(-1, vocab_size)
            if after_outs is not None:
                after_outs = after_outs.view(-1, vocab_size)
            ys = ys.view(-1)

        # calculate loss
        ce_loss = self.ce_criterion(before_outs, ys)
        before_acc = (before_outs.argmax(-1) == ys).sum() / len(ys)
        if after_outs is not None:
            ce_loss += self.ce_criterion(after_outs, ys)
            after_acc = (after_outs.argmax(-1) == ys).sum() / len(ys)
        else:
            after_acc = None
        duration_loss = self.duration_criterion(d_outs, ds)
        pitch_loss = self.mse_criterion(p_outs, ps)
        energy_loss = self.mse_criterion(e_outs, es)

        # make weighted mask and apply it
        if self.use_weighted_masking:
            ce_loss = ce_loss.view(batch_size, max_len)
            out_masks = make_non_pad_mask(olens).to(ys.device)
            out_weights = out_masks.float() / out_masks.sum(dim=1, keepdim=True).float()
            out_weights /= batch_size
            duration_masks = make_non_pad_mask(ilens).to(ys.device)
            duration_weights = (
                duration_masks.float() / duration_masks.sum(dim=1, keepdim=True).float()
            )
            duration_weights /= ds.size(0)

            # apply weight
            ce_loss = ce_loss.mul(out_weights).masked_select(out_masks).sum()
            duration_loss = (
                duration_loss.mul(duration_weights).masked_select(duration_masks).sum()
            )
            pitch_masks = duration_masks.unsqueeze(-1)
            pitch_weights = duration_weights.unsqueeze(-1)
            pitch_loss = pitch_loss.mul(pitch_weights).masked_select(pitch_masks).sum()
            energy_loss = (
                energy_loss.mul(pitch_weights).masked_select(pitch_masks).sum()
            )

        return ce_loss, duration_loss, pitch_loss, energy_loss, before_acc, after_acc
