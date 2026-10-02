# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch


def masked_mse_loss(input, target, mask, weights=None):
    """Average squared errors over observed labels, with optional label weights."""
    errors = (input.masked_select(mask) - target.masked_select(mask)) ** 2
    if weights is not None:
        errors = errors * weights.expand_as(input).masked_select(mask)
    return errors.sum() / mask.sum().clamp(min=1)


def masked_l1_loss(input, target, mask, weights=None):
    """Average absolute errors over observed labels, with optional label weights."""
    errors = torch.abs(input.masked_select(mask) - target.masked_select(mask))
    if weights is not None:
        errors = errors * weights.expand_as(input).masked_select(mask)
    return errors.sum() / mask.sum().clamp(min=1)
