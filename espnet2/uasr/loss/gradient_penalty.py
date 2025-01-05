import numpy as np
import torch
from torch import autograd
from typeguard import typechecked

from espnet2.uasr.discriminator.abs_discriminator import AbsDiscriminator
from espnet2.uasr.loss.abs_loss import AbsUASRLoss
from espnet2.utils.types import str2bool


class UASRGradientPenalty(AbsUASRLoss):
    """
    Gradient penalty for Unsupervised Audio Speech Recognition (UASR).

    This class implements a gradient penalty mechanism to be used in the
    training of a discriminator in UASR tasks. The gradient penalty helps
    to enforce Lipschitz continuity, which is crucial for training
    Generative Adversarial Networks (GANs) effectively.

    Attributes:
        discriminator (List[AbsDiscriminator]): The discriminator model.
        weight (float): The weight applied to the gradient penalty.
        probabilistic_grad_penalty_slicing (bool): If True, uses probabilistic
            slicing for samples.
        reduction (str): Specifies the reduction method to apply to the
            output ('sum' or 'mean').

    Args:
        discriminator (AbsDiscriminator): The discriminator model.
        weight (float): Weight for the gradient penalty (default is 1.0).
        probabilistic_grad_penalty_slicing (str2bool): Flag for probabilistic
            gradient penalty slicing (default is False).
        reduction (str): Reduction method ('sum' or 'mean', default is 'sum').

    Returns:
        torch.Tensor: The computed gradient penalty value.

    Examples:
        >>> discriminator = SomeDiscriminatorModel()
        >>> loss_fn = UASRGradientPenalty(discriminator, weight=10.0)
        >>> fake_samples = torch.randn(16, 100)  # Batch of fake samples
        >>> real_samples = torch.randn(16, 100)  # Batch of real samples
        >>> loss = loss_fn(fake_samples, real_samples, True, True)
        >>> print(loss)

    Note:
        The `is_training` argument should be set to True when the model is
        in training mode, and `is_discrimininative_step` should be set to
        True when the discriminator is being trained.

    Raises:
        ValueError: If the shapes of `fake_sample` and `real_sample` do not
            match.

    Todo:
        - Implement additional slicing strategies for better performance.
    """

    @typechecked
    def __init__(
        self,
        discriminator: AbsDiscriminator,
        weight: float = 1.0,
        probabilistic_grad_penalty_slicing: str2bool = False,
        reduction: str = "sum",
    ):
        super().__init__()

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
        """
        Computes the gradient penalty for UASR during training.

        This method calculates the gradient penalty as part of the UASR loss
        function, which is crucial for training the discriminator in a GAN-like
        setup. The gradient penalty helps enforce the Lipschitz constraint,
        ensuring the model is more stable during training.

        Args:
            fake_sample (torch.Tensor): A tensor representing the generated
                sample from the generator.
            real_sample (torch.Tensor): A tensor representing the real sample.
            is_training (str2bool): A boolean indicating whether the model
                is currently in the training phase.
            is_discrimininative_step (str2bool): A boolean indicating whether
                the current step is focused on training the discriminator.

        Returns:
            torch.Tensor: The computed gradient penalty value, which is the
            sum of the squared norm of the gradients, or zero if the
            conditions for computing the penalty are not met.

        Note:
            The method uses probabilistic slicing if
            `probabilistic_grad_penalty_slicing` is set to True. Otherwise,
            it slices the samples based on the batch size and time length
            directly.

        Examples:
            >>> fake_samples = torch.randn(32, 100)  # Batch of fake samples
            >>> real_samples = torch.randn(32, 100)  # Batch of real samples
            >>> loss = uasr_gradient_penalty.forward(
            ...     fake_samples, real_samples,
            ...     is_training=True, is_discrimininative_step=True)
            >>> print(loss)
        """
        if self.weight > 0 and is_discrimininative_step and is_training:
            batch_size = min(fake_sample.size(0), real_sample.size(0))
            time_length = min(fake_sample.size(1), real_sample.size(1))

            if self.probabilistic_grad_penalty_slicing:

                def get_slice(sample, dim, target_size):
                    size = sample.size(dim)
                    diff = size - target_size
                    if diff <= 0:
                        return sample

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
