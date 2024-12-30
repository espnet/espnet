# Copyright 2022 Dan Lim
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""JETS related loss module for ESPnet2."""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet.nets.pytorch_backend.fastspeech.duration_predictor import (  # noqa: H301
    DurationPredictorLoss,
)
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask


class VarianceLoss(torch.nn.Module):
    """
        VarianceLoss is a PyTorch module that computes the variance loss for JETS,
    which includes duration, pitch, and energy predictions. It allows for masking
    of padded sequences and supports weighted masking to adjust the loss based on
    the importance of different input elements.

    Attributes:
        use_masking (bool): Flag indicating whether to apply masking to the loss
            calculation.
        use_weighted_masking (bool): Flag indicating whether to use weighted
            masking in the loss calculation.
        mse_criterion (torch.nn.MSELoss): Mean Squared Error loss function.
        duration_criterion (DurationPredictorLoss): Duration prediction loss
            function.

    Args:
        use_masking (bool): Whether to apply masking for padded part in loss
            calculation. Defaults to True.
        use_weighted_masking (bool): Whether to apply weighted masking in loss
            calculation. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The calculated loss values
            for duration, pitch, and energy predictions.

    Examples:
        >>> variance_loss = VarianceLoss(use_masking=True, use_weighted_masking=False)
        >>> d_outs = torch.randn(2, 5)  # Example duration outputs
        >>> ds = torch.randint(1, 6, (2, 5))  # Example durations
        >>> p_outs = torch.randn(2, 5, 1)  # Example pitch outputs
        >>> ps = torch.randn(2, 5, 1)  # Example target pitch
        >>> e_outs = torch.randn(2, 5, 1)  # Example energy outputs
        >>> es = torch.randn(2, 5, 1)  # Example target energy
        >>> ilens = torch.tensor([5, 5])  # Example input lengths
        >>> duration_loss, pitch_loss, energy_loss = variance_loss(d_outs, ds, p_outs, ps, e_outs, es, ilens)

    Note:
        The variance loss is specifically designed for the JETS framework in
        ESPnet2 and should be used in conjunction with other components of the
        JETS model.
    """

    @typechecked
    def __init__(self, use_masking: bool = True, use_weighted_masking: bool = False):
        """Initialize JETS variance loss module.

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
        self.mse_criterion = torch.nn.MSELoss(reduction=reduction)
        self.duration_criterion = DurationPredictorLoss(reduction=reduction)

    def forward(
        self,
        d_outs: torch.Tensor,
        ds: torch.Tensor,
        p_outs: torch.Tensor,
        ps: torch.Tensor,
        e_outs: torch.Tensor,
        es: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate forward propagation.

        This method computes the variance loss for the duration, pitch, and
        energy predictors based on the provided outputs and targets. It applies
        masking to ignore padded parts of the input during loss calculation if
        specified.

        Args:
            d_outs (LongTensor): Batch of outputs of duration predictor
                (B, T_text).
            ds (LongTensor): Batch of durations (B, T_text).
            p_outs (Tensor): Batch of outputs of pitch predictor (B, T_text, 1).
            ps (Tensor): Batch of target token-averaged pitch (B, T_text, 1).
            e_outs (Tensor): Batch of outputs of energy predictor (B, T_text, 1).
            es (Tensor): Batch of target token-averaged energy (B, T_text, 1).
            ilens (LongTensor): Batch of the lengths of each input (B,).

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: A tuple containing the
            duration predictor loss value, pitch predictor loss value, and
            energy predictor loss value.

        Examples:
            >>> variance_loss = VarianceLoss(use_masking=True)
            >>> d_outs = torch.randn(2, 5)
            >>> ds = torch.randint(1, 10, (2, 5))
            >>> p_outs = torch.randn(2, 5, 1)
            >>> ps = torch.randn(2, 5, 1)
            >>> e_outs = torch.randn(2, 5, 1)
            >>> es = torch.randn(2, 5, 1)
            >>> ilens = torch.tensor([5, 5])
            >>> losses = variance_loss(d_outs, ds, p_outs, ps, e_outs, es, ilens)
            >>> print(losses)
        """
        # apply mask to remove padded part
        if self.use_masking:
            duration_masks = make_non_pad_mask(ilens).to(ds.device)
            d_outs = d_outs.masked_select(duration_masks)
            ds = ds.masked_select(duration_masks)
            pitch_masks = make_non_pad_mask(ilens).unsqueeze(-1).to(ds.device)
            p_outs = p_outs.masked_select(pitch_masks)
            e_outs = e_outs.masked_select(pitch_masks)
            ps = ps.masked_select(pitch_masks)
            es = es.masked_select(pitch_masks)

        # calculate loss
        duration_loss = self.duration_criterion(d_outs, ds)
        pitch_loss = self.mse_criterion(p_outs, ps)
        energy_loss = self.mse_criterion(e_outs, es)

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
            pitch_masks = duration_masks.unsqueeze(-1)
            pitch_weights = duration_weights.unsqueeze(-1)
            pitch_loss = pitch_loss.mul(pitch_weights).masked_select(pitch_masks).sum()
            energy_loss = (
                energy_loss.mul(pitch_weights).masked_select(pitch_masks).sum()
            )

        return duration_loss, pitch_loss, energy_loss


class ForwardSumLoss(torch.nn.Module):
    """
        Forwardsum loss described at https://openreview.net/forum?id=0NQwnnwAORi

    This module computes the forwardsum loss for a batch of input sequences.
    The forwardsum loss is particularly useful in tasks where the alignment
    between input and target sequences is crucial, such as in speech
    recognition or text-to-speech systems.

    Attributes:
        None

    Args:
        log_p_attn (Tensor): Batch of log probability of attention matrix
            (B, T_feats, T_text).
        ilens (Tensor): Batch of the lengths of each input (B,).
        olens (Tensor): Batch of the lengths of each target (B,).
        blank_prob (float): Blank symbol probability. Default is e^-1.

    Returns:
        Tensor: forwardsum loss value.

    Examples:
        >>> model = ForwardSumLoss()
        >>> log_p_attn = torch.randn(4, 10, 20)  # Example log probabilities
        >>> ilens = torch.tensor([10, 9, 8, 7])   # Input lengths
        >>> olens = torch.tensor([5, 4, 3, 2])    # Output lengths
        >>> loss = model(log_p_attn, ilens, olens)
        >>> print(loss)
    """

    def __init__(self):
        """Initialize forwardsum loss module."""
        super().__init__()

    def forward(
        self,
        log_p_attn: torch.Tensor,
        ilens: torch.Tensor,
        olens: torch.Tensor,
        blank_prob: float = np.e**-1,
    ) -> torch.Tensor:
        """
                Forwardsum loss described at https://openreview.net/forum?id=0NQwnnwAORi

        Attributes:
            None

        Args:
            log_p_attn (Tensor): Batch of log probability of attention matrix
                (B, T_feats, T_text).
            ilens (Tensor): Batch of the lengths of each input (B,).
            olens (Tensor): Batch of the lengths of each target (B,).
            blank_prob (float): Blank symbol probability. Default is e^-1.

        Returns:
            Tensor: forwardsum loss value.

        Examples:
            >>> model = ForwardSumLoss()
            >>> log_p_attn = torch.randn(2, 5, 4)  # Example log probabilities
            >>> ilens = torch.tensor([4, 3])
            >>> olens = torch.tensor([3, 2])
            >>> loss = model(log_p_attn, ilens, olens)
            >>> print(loss)
        """
        B = log_p_attn.size(0)

        # a row must be added to the attention matrix to account for
        #    blank token of CTC loss
        # (B,T_feats,T_text+1)
        log_p_attn_pd = F.pad(log_p_attn, (1, 0, 0, 0, 0, 0), value=np.log(blank_prob))

        loss = 0
        for bidx in range(B):
            # construct target sequnece.
            # Every text token is mapped to a unique sequnece number.
            target_seq = torch.arange(1, ilens[bidx] + 1).unsqueeze(0)
            cur_log_p_attn_pd = log_p_attn_pd[
                bidx, : olens[bidx], : ilens[bidx] + 1
            ].unsqueeze(
                1
            )  # (T_feats,1,T_text+1)
            cur_log_p_attn_pd = F.log_softmax(cur_log_p_attn_pd, dim=-1)
            loss += F.ctc_loss(
                log_probs=cur_log_p_attn_pd,
                targets=target_seq,
                input_lengths=olens[bidx : bidx + 1],
                target_lengths=ilens[bidx : bidx + 1],
                zero_infinity=True,
            )
        loss = loss / B
        return loss
