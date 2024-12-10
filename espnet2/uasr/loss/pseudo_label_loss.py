import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.uasr.loss.abs_loss import AbsUASRLoss
from espnet2.utils.types import str2bool


class UASRPseudoLabelLoss(AbsUASRLoss):
    """
    Auxiliary pseudo label loss for Unsupervised Automatic Speech Recognition (UASR).

    This loss function computes the pseudo label loss using the cross-entropy
    between the model's output and the provided pseudo labels. It is designed
    to be used in scenarios where labeled data is scarce, allowing the model
    to learn from its own predictions.

    Attributes:
        weight (float): Weight of the loss. If set to 0, the loss will not
            contribute to the overall loss.
        input_dim (int): Dimensionality of the input features.
        output_dim (int): Dimensionality of the output features.
        downsample_rate (int): Rate at which to downsample the pseudo labels.
        ignore_index (int): Index that is ignored and does not contribute to
            the loss computation.
        reduction (str): Specifies the reduction to apply to the output.
            Options are 'none', 'mean', 'sum'.

    Args:
        weight (float): Weight of the loss. Default is 1.0.
        input_dim (int): Dimensionality of the input features. Default is 128.
        output_dim (int): Dimensionality of the output features. Default is 64.
        downsample_rate (int): Rate at which to downsample the pseudo labels.
            Default is 2.
        ignore_index (int): Index that is ignored in the loss computation.
            Default is -1.
        reduction (str): Specifies the reduction method. Default is 'none'.

    Returns:
        torch.Tensor: The computed pseudo label loss.

    Examples:
        >>> loss_fn = UASRPseudoLabelLoss(weight=1.0)
        >>> inter_x = torch.randn(10, 20, 128)  # Batch of 10, seq length 20, input_dim 128
        >>> pseudo_labels = torch.randint(0, 64, (10, 10))  # Batch of 10, seq length 10
        >>> is_discriminative_step = False
        >>> loss = loss_fn(inter_x, pseudo_labels, is_discriminative_step)
        >>> print(loss)

    Note:
        The loss will return 0 if the weight is 0 or if it is in a
        discriminative step.

    Todo:
        - Add support for additional reduction methods if needed.
    """

    @typechecked
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
        """
        Forward pass for calculating the pseudo label loss.

        This method computes the pseudo label loss based on the input tensor
        and the provided pseudo labels. The loss is calculated using cross
        entropy, taking into account the downsampling rate and any specified
        parameters such as the ignore index and reduction method.

        Args:
            inter_x (torch.Tensor): The input tensor from the model's
                intermediate layer. Shape should be (batch_size, time_steps,
                input_dim).
            pseudo_labels (torch.Tensor): The tensor containing the pseudo
                labels for the input. Shape should be (batch_size, time_steps).
            is_discriminative_step (str2bool): A boolean indicating whether
                the current step is discriminative. If False, the loss will
                be calculated.

        Returns:
            torch.Tensor: The computed pseudo label loss. Returns 0 if the
            weight is 0, if the step is discriminative, or if pseudo labels
            are None.

        Examples:
            >>> loss_fn = UASRPseudoLabelLoss(weight=1.0)
            >>> inter_x = torch.randn(10, 20, 128)  # Batch of 10, 20 time steps
            >>> pseudo_labels = torch.randint(0, 64, (10, 10))  # Pseudo labels
            >>> is_discriminative_step = False
            >>> loss = loss_fn(inter_x, pseudo_labels, is_discriminative_step)
            >>> print(loss)

        Note:
            Ensure that `inter_x` and `pseudo_labels` are appropriately shaped
            and that the `weight` parameter is set according to the desired
            importance of the pseudo label loss in the overall loss computation.
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
