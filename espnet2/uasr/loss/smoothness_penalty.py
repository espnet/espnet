import torch
import torch.nn.functional as F

from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet2.uasr.loss.abs_loss import AbsUASRLoss
from espnet2.utils.types import str2bool


class UASRSmoothnessPenalty(AbsUASRLoss):
    """smoothness penalty for UASR."""

    def __init__(
        self,
        weight: float = 1.0,
        reduction: str = "none",
    ):
        super().__init__()
        assert check_argument_types()

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
