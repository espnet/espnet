# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import torch.nn.functional as F


def masked_mse_loss(input, target, mask):
    """
    Compute masked mean squared error (MSE) loss.

    Args:
        input (torch.Tensor): Input tensor.
        target (torch.Tensor): Target tensor.
        mask (torch.Tensor): Mask tensor of the same shape as input and target.

    Returns:
        torch.Tensor: Masked MSE loss.
    """
    squared_diff = (input.masked_select(mask) - target.masked_select(mask)) ** 2
    loss = torch.mean(squared_diff)
    return loss


def masked_l1_loss(input, target, mask):
    """
    Compute masked L1 loss.

    Args:
        input (torch.Tensor): Input tensor.
        target (torch.Tensor): Target tensor.
        mask (torch.Tensor): Mask tensor of the same shape as input and target.

    Returns:
        torch.Tensor: Masked L1 loss.
    """
    abs_diff = torch.abs(input.masked_select(mask) - target.masked_select(mask))
    loss = torch.mean(abs_diff)
    return loss
