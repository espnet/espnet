import torch
import torch.nn.functional as F

from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import to_device, make_pad_mask
from espnet2.uasr.loss.abs_loss import AbsUASRLoss
from espnet2.utils.types import str2bool


class UASRAutoencoderlLoss(AbsUASRLoss):
    """autoencoder loss for UASR."""

    def __init__(
        self,
        weight: float = 1.0,
        input_dim: int = 64,
        output_dim: int = 100,
        ignore_index: int = -1,
        reduction: str = "none",
    ):
        super().__init__()
        assert check_argument_types()

        self.weight = weight
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ignore_index = ignore_index
        self.reduction = reduction

        if self.weight > 0:
            self.decoder = torch.nn.Linear(self.input_dim, self.output_dim)

    def forward(
        self,
        generated_sample: torch.Tensor,
        generated_sample_padding_mask: torch.Tensor,
        input_cluster_id: torch.Tensor,
        input_cluster_id_lengths: torch.Tensor,
        is_discriminative_step: str2bool,
    ):
        """Forward.

        Args:
        """
        if (
            self.weight > 0
            and not is_discriminative_step
            and input_cluster_id is not None
        ):
            autoencoder_prediction = self.decoder(generated_sample)
            cluster_id_mask = make_pad_mask(input_cluster_id_lengths)
            input_cluster_id[cluster_id_mask] = self.ignore_index

            sync_length = min(autoencoder_prediction.size(1), input_cluster_id.size(1))

            autoencoder_loss = F.cross_entropy(
                autoencoder_prediction[:, :sync_length].transpose(1, 2),
                input_cluster_id[:, :sync_length],
                ignore_index=self.ignore_index,
                reduction=self.reduction,
            )
            autoencoder_loss = autoencoder_loss.mean() * autoencoder_loss.shape[0]

            return autoencoder_loss
        else:
            return 0
