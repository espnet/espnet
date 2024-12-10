import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.uasr.loss.abs_loss import AbsUASRLoss


class UASRSmoothnessPenalty(AbsUASRLoss):
    """
    Smoothness penalty for Unsupervised ASR (UASR).

    This class implements a smoothness penalty loss that encourages the output
    logits of the model to be smooth over time. It is typically used during
    training to promote continuity in the predicted outputs.

    Attributes:
        weight (float): The weight of the smoothness penalty. Default is 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Options are 'none', 'mean', 'sum'. Default is 'none'.

    Args:
        weight (float): Weight of the smoothness penalty. Default is 1.0.
        reduction (str): Specifies the reduction method. Default is 'none'.

    Returns:
        torch.Tensor: The computed smoothness penalty.

    Examples:
        >>> smoothness_penalty = UASRSmoothnessPenalty(weight=0.5)
        >>> logits = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        >>> padding_mask = torch.tensor([[False, False, True],
        ...                               [False, False, False]])
        >>> penalty = smoothness_penalty(
        ...     dense_logits=logits,
        ...     dense_padding_mask=padding_mask,
        ...     sample_size=2,
        ...     is_discriminative_step=False
        ... )
        >>> print(penalty)

    Note:
        The penalty is only computed if the weight is greater than 0 and the
        current step is not a discriminative step.

    Raises:
        ValueError: If the reduction method is not one of 'none', 'mean', or 'sum'.
    """

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
        """
        Computes the smoothness penalty for the UASR (Unsupervised Automatic Speech
        Recognition) task.

        This method calculates a smoothness penalty based on the output logits of
        the generator. The smoothness penalty is computed using the mean squared
        error (MSE) between adjacent logits, with the option to apply a reduction
        method. If the weight is set to zero or if it is a discriminative step,
        the penalty will be zero.

        Args:
            dense_logits (torch.Tensor): Output logits of the generator.
            dense_padding_mask (torch.Tensor): Padding mask of logits.
            sample_size (int): Batch size.
            is_discriminative_step (bool): Indicates whether it is a training
                step for the discriminator.

        Returns:
            torch.Tensor: The computed smoothness penalty. Returns 0 if the
            weight is 0 or if it is a discriminative step.

        Examples:
            >>> import torch
            >>> penalty = UASRSmoothnessPenalty(weight=1.0)
            >>> logits = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
            >>> padding_mask = torch.tensor([[False, False, False],
            ...                               [False, False, True]])
            >>> sample_size = 2
            >>> is_discriminative_step = False
            >>> result = penalty.forward(logits, padding_mask, sample_size,
            ...                           is_discriminative_step)
            >>> print(result)

        Note:
            The `reduction` attribute can be set to control the way the penalty
            is computed. It can be set to 'none', 'mean', or 'sum'.
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
