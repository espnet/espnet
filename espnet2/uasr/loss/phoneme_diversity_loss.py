import torch
from typeguard import typechecked

from espnet2.uasr.loss.abs_loss import AbsUASRLoss
from espnet2.utils.types import str2bool


class UASRPhonemeDiversityLoss(AbsUASRLoss):
    """phoneme diversity loss for UASR."""

    @typechecked
    def __init__(
        self,
        weight: float = 1.0,
    ):
        super().__init__()

        self.weight = weight

    def forward(
        self, dense_x: torch.Tensor, sample_size: int, is_discriminative_step: str2bool
    ):
        """Forward.

        Args:
            dense_x: predicted logits of generated samples
            sample_size: batch size
            is_dicriminative_step: whether is training discriminator
        """
        if self.weight > 0 and not is_discriminative_step:
            batch_size, time_length, channel_size = dense_x.shape

            avg_probs = torch.softmax(
                dense_x.reshape(-1, channel_size).float(), dim=-1
            ).mean(dim=0)
            phoneme_ppl = torch.exp(
                -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
            )
            phoneme_diversity_loss = (
                (channel_size - phoneme_ppl) / channel_size
            ) * sample_size

            return phoneme_diversity_loss
        else:
            return 0
