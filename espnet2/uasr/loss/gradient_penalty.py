import torch
from torch import autograd
from typeguard import check_argument_types

from espnet2.uasr.discriminator.abs_discriminator import AbsDiscriminator
from espnet2.uasr.loss.abs_loss import AbsUASRLoss
from espnet2.utils.types import str2bool
from espnet.nets.pytorch_backend.nets_utils import to_device


class UASRGradientPenalty(AbsUASRLoss):
    """gradient penalty for UASR."""

    def __init__(
        self,
        discriminator: AbsDiscriminator,
        weight: float = 1.0,
        probabilistic_grad_penalty_slicing: str2bool = False,
        reduction: str = "sum",
    ):
        super().__init__()
        assert check_argument_types()

        self.discriminator = [discriminator]
        self.weight = weight
        self.probabilistic_grad_penalty_slicing = probabilistic_grad_penalty_slicing
        self.reduction = reduction

    def forward(
        self,
        fake_sample: torch.Tensor,
        real_sample: torch.Tensor,
        is_training: str2bool,
        is_discrimininative_step: str2bool,
    ):
        """Forward.

        Args:
            fake_sample: generated sample from generator
            real_sample: real sample
            is_training: whether is at training step
            is_discriminative_step: whether is training discriminator
        """
        if self.weight > 0 and is_discrimininative_step and is_training:
            batch_size = min(fake_sample.size(0), real_sample.size(0))
            time_length = min(fake_sample.size(1), real_sample.size(1))

            if self.probabilistic_grad_penalty_slicing:

                def get_slice(sample, dim, target_size):
                    size = sample.size(dim)
                    diff = size - target_size
                    if diff <= 0:
                        return data

                    start = np.random.randint(0, diff + 1)
                    return sample.narrow(dim=dim, start=start, length=target_size)

                fake_sample = get_slice(fake_sample, 0, batch_size)
                fake_sample = get_slice(fake_sample, 1, time_length)
                real_sample = get_slice(real_sample, 0, batch_size)
                real_sample = get_slice(real_sample, 1, time_length)

            else:
                fake_sample = fake_sample[:batch_size, :time_length]
                real_sample = real_sample[:batch_size, :time_length]

            alpha = torch.rand(real_sample.size(0), 1, 1)
            alpha = alpha.expand(real_sample.size())
            alpha = alpha.to(real_sample.device)

            interpolates = alpha * real_sample + ((1 - alpha) * fake_sample)

            disc_interpolates = self.discriminator[0](interpolates, None)

            gradients = autograd.grad(
                outputs=disc_interpolates,
                inputs=interpolates,
                grad_outputs=torch.ones(
                    disc_interpolates.size(), device=real_sample.device
                ),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            gradient_penalty = (gradients.norm(2, dim=1) - 1) ** 2
            return gradient_penalty.sum()
        else:
            return 0
