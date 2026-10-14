"""Compatibility imports; use espnet2.aqa.base.loss for new code."""

from espnet2.aqa.base.loss import (  # noqa: F401
    masked_l1_loss,
    masked_mse_loss,
)
