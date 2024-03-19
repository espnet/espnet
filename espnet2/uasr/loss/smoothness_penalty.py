import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.uasr.loss.abs_loss import AbsUASRLoss


class UASRSmoothnessPenalty(AbsUASRLoss):
    """smoothness penalty for UASR."""

    @typechecked
    def __init__(
        self,
        weight: float = 1.0,
        reduction: str = "none",
    ):
        super().__init__()

        self.weight = weight
        self.reduction = reduction

    def forward(
        self,
        dense_logits: torch.Tensor,
        dense_padding_mask: torch.Tensor,
        sample_size: int,
        is_discriminative_step: bool,
    ):
        """Forward.

        Args:
            dense_logits: output logits of generator
            dense_padding_mask: padding mask of logits
            sample_size: batch size
            is_discriminative_step: Whether is training discriminator
        """
        if self.weight > 0 and not is_discriminative_step:
            smoothness_penalty = F.mse_loss(
                dense_logits[:, :-1], dense_logits[:, 1:], reduction=self.reduction
            )
            smoothness_penalty[dense_padding_mask[:, 1:]] = 0
            smoothness_penalty = smoothness_penalty.mean() * sample_size

            return smoothness_penalty
        else:
            return 0
