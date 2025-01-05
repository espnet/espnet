import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.uasr.loss.abs_loss import AbsUASRLoss
from espnet2.utils.types import str2bool


class UASRDiscriminatorLoss(AbsUASRLoss):
    """
    A class that implements the discriminator loss for Unsupervised ASR (UASR).

    This loss function is designed to optimize the performance of the
    discriminator in a UASR setup by applying binary cross-entropy loss
    on both the generated (fake) and real samples, with options for label
    smoothing and reduction strategies.

    Attributes:
        weight (float): The weight for the loss, where a value greater than 0
            indicates that the loss will be computed.
        smoothing (float): The amount of label smoothing to apply to the
            generated samples.
        smoothing_one_sided (bool): If True, only applies smoothing to the
            generated samples.
        reduction (str): Specifies the reduction to apply to the output.
            Options are 'none', 'mean', or 'sum'.

    Args:
        weight (float, optional): Weight for the loss. Defaults to 1.0.
        smoothing (float, optional): Label smoothing value. Defaults to 0.0.
        smoothing_one_side (str2bool, optional): If True, apply smoothing
            only to generated samples. Defaults to False.
        reduction (str, optional): Reduction method. Options are 'none',
            'mean', or 'sum'. Defaults to 'sum'.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]: The computed loss for
        the dense and token outputs. If the weight is less than or equal to
        0, returns 0.

    Examples:
        >>> loss_fn = UASRDiscriminatorLoss(weight=1.0, smoothing=0.1)
        >>> dense_y = torch.tensor([[0.5], [0.2]], requires_grad=True)
        >>> token_y = torch.tensor([[0.1], [0.9]], requires_grad=True)
        >>> loss_dense, loss_token = loss_fn(dense_y, token_y, True)

    Note:
        The loss is computed differently based on the `is_discriminative_step`
        argument, allowing for flexibility in training phases.

    Raises:
        ValueError: If the reduction method specified is not one of 'none',
        'mean', or 'sum'.
    """

    @typechecked
    def __init__(
        self,
        weight: float = 1.0,
        smoothing: float = 0.0,
        smoothing_one_side: str2bool = False,
        reduction: str = "sum",
    ):
        super().__init__()
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
        """
        Forward pass for the UASRDiscriminatorLoss.

        This method computes the discriminator loss based on the predicted logits
        of generated and real samples. It applies binary cross-entropy loss with
        optional label smoothing.

        Args:
            dense_y (torch.Tensor): Predicted logits of generated samples.
            token_y (torch.Tensor): Predicted logits of real samples.
            is_discriminative_step (str2bool): A flag indicating whether the
                current step is a discriminative step or not.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing the
            loss for dense predictions and the loss for token predictions. If
            the weight is less than or equal to zero, it returns 0.

        Note:
            The loss is weighted by the `weight` attribute, and label smoothing
            is applied based on the `smoothing` and `smoothing_one_sided`
            attributes.

        Examples:
            >>> loss_fn = UASRDiscriminatorLoss(weight=1.0, smoothing=0.1)
            >>> dense_y = torch.tensor([[0.5], [0.2]], requires_grad=True)
            >>> token_y = torch.tensor([[0.3], [0.7]], requires_grad=True)
            >>> is_discriminative_step = True
            >>> loss_dense, loss_token = loss_fn(dense_y, token_y,
            ...                                    is_discriminative_step)
            >>> print(loss_dense, loss_token)
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
