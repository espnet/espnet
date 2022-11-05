import torch
import torch.nn.functional as F

from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet2.uasr.loss.abs_loss import AbsUASRLoss
from espnet2.utils.types import str2bool


class UASRDiscriminatorLoss(AbsUASRLoss):
    """discriminator loss for UASR."""

    def __init__(
        self,
        weight: float = 1.0,
        smoothing: float = 0.0,
        smoothing_one_side: str2bool = False,
        reduction: str = "sum",
    ):
        super().__init__()
        assert check_argument_types()
        self.weight = weight
        self.smoothing = smoothing
        self.smoothing_one_sided = smoothing_one_side
        self.reduction = reduction

    def forward(
        self,
        dense_y: torch.Tensor,
        token_y: torch.Tensor,
        is_discriminative_step: str2bool,
    ):
        """Forward.

        Args:
        """
        if self.weight > 0:
            fake_smooth = self.smoothing
            real_smooth = self.smoothing
            if self.smoothing_one_sided:
                fake_smooth = 0

            if is_discriminative_step:
                loss_dense = F.binary_cross_entropy_with_logits(
                    dense_y,
                    dense_y.new_ones(dense_y.shape) - fake_smooth,
                    reduction=self.reduction,
                )
                loss_token = F.binary_cross_entropy_with_logits(
                    token_y,
                    token_y.new_zeros(token_y.shape) + real_smooth,
                    reduction=self.reduction,
                )
            else:
                loss_dense = F.binary_cross_entropy_with_logits(
                    dense_y,
                    dense_y.new_zeros(dense_y.shape) + fake_smooth,
                    reduction=self.reduction,
                )
                loss_token = None

            return loss_dense, loss_token
        else:
            return 0
