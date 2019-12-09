#!/usr/bin/env python3

"""Transducer loss module."""

from torch import nn

from warprnnt_pytorch import RNNTLoss


class TransLoss(nn.Module):
    """Transducer loss.

    Args:
        trans_type (str): type of transducer implementation to calculate loss.
        blank_id (int): blank symbol id

    """

    def __init__(self, trans_type, blank_id):
        """Construct an TransLoss object."""
        super(TransLoss, self).__init__()

        if trans_type == 'warp-transducer':
            self.trans_loss = RNNTLoss(blank=blank_id)
        else:
            raise NotImplementedError

        self.blank_id = blank_id

    def forward(self, pred_pad, target, pred_len, target_len):
        """Compute path-aware regularization transducer loss.

        Args:
            pred_pad (torch.Tensor): Batch of predicted sequence (B, T, U+1, odim)
            target_pad (torch.Tensor): Batch of target sequence (B, U)
            hs_pad (torch.Tensor): batch of padded input sequences (B, Tmax)
            ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)
            hlens (torch.tensor): batch of lengths of input sequences (B)

        Returns:
            loss (torch.Tensor): transducer loss

        """
        loss = self.trans_loss(pred_pad, target, pred_len, target_len)

        return loss
