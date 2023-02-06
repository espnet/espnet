import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet2.uasr.loss.abs_loss import AbsUASRLoss
from espnet2.utils.types import str2bool


class UASRPseudoLabelLoss(AbsUASRLoss):
    """auxiliary pseudo label loss for UASR."""

    def __init__(
        self,
        weight: float = 1.0,
        input_dim: int = 128,
        output_dim: int = 64,
        downsample_rate: int = 2,
        ignore_index: int = -1,
        reduction: str = "none",
    ):
        super().__init__()
        assert check_argument_types()

        self.weight = weight
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.downsample_rate = downsample_rate
        self.ignore_index = ignore_index
        self.reduction = reduction

        if self.weight > 0:
            self.decoder = torch.nn.Linear(self.input_dim, self.output_dim)

    def forward(
        self,
        inter_x: torch.Tensor,
        pseudo_labels: torch.Tensor,
        is_discriminative_step: str2bool,
    ):
        """Forward.

        Args:
        """
        if self.weight > 0 and not is_discriminative_step and pseudo_labels is not None:
            inter_x = self.decoder(inter_x)

            if self.downsample_rate > 1:
                pseudo_labels = pseudo_labels[:, :: self.downsample_rate]
            valid_time_length = min(pseudo_labels.shape[1], inter_x.shape[1])
            pseudo_label_loss = F.cross_entropy(
                inter_x[:, :valid_time_length].transpose(1, 2),
                pseudo_labels[:, :valid_time_length],
                ignore_index=self.ignore_index,
                reduction=self.reduction,
            )
            pseudo_label_loss = pseudo_label_loss.mean() * pseudo_label_loss.shape[0]

            return pseudo_label_loss
        else:
            return 0
