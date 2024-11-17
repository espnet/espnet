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
    squared_diff = (input - target) ** 2
    masked_squared_diff = squared_diff * mask
    loss = torch.sum(masked_squared_diff) / torch.sum(mask)
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
    abs_diff = torch.abs(input - target)
    masked_abs_diff = abs_diff * mask
    loss = torch.sum(masked_abs_diff) / torch.sum(mask)
    return loss
