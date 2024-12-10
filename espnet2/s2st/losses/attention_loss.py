import torch
from typeguard import typechecked

from espnet2.s2st.losses.abs_loss import AbsS2STLoss
from espnet2.utils.types import str2bool
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (  # noqa: H301
    LabelSmoothingLoss,
)


class S2STAttentionLoss(AbsS2STLoss):
    """
    attention-based label smoothing loss for S2ST.

    This class implements an attention-based label smoothing loss specifically
    designed for sequence-to-sequence tasks. It utilizes label smoothing to
    improve the robustness of the model against label noise and overfitting.

    Attributes:
        weight (float): The weight for the loss function.
        loss (LabelSmoothingLoss): The label smoothing loss instance.

    Args:
        vocab_size (int): The size of the vocabulary.
        padding_idx (int, optional): The index used for padding. Defaults to -1.
        weight (float, optional): The weight for the loss function. Defaults to 1.0.
        smoothing (float, optional): The label smoothing factor. Defaults to 0.0.
        normalize_length (str2bool, optional): Whether to normalize the loss by
            the length of the sequence. Defaults to False.
        criterion (torch.nn.Module, optional): The criterion used for the loss.
            Defaults to torch.nn.KLDivLoss(reduction="none").

    Returns:
        torch.Tensor or None: The computed loss value if weight is greater than
        0; otherwise, returns None.

    Examples:
        >>> loss_fn = S2STAttentionLoss(vocab_size=1000)
        >>> dense_y = torch.randn(10, 1000)  # Example output probabilities
        >>> token_y = torch.randint(0, 1000, (10,))  # Example target tokens
        >>> loss = loss_fn(dense_y, token_y)
        >>> print(loss)

    Note:
        Ensure that the dense_y tensor is properly normalized before passing it
        to the loss function.
    """

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        padding_idx: int = -1,
        weight: float = 1.0,
        smoothing: float = 0.0,
        normalize_length: str2bool = False,
        criterion: torch.nn.Module = torch.nn.KLDivLoss(reduction="none"),
    ):
        super().__init__()
        self.weight = weight
        self.loss = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=padding_idx,
            smoothing=smoothing,
            normalize_length=normalize_length,
            criterion=criterion,
        )

    def forward(
        self,
        dense_y: torch.Tensor,
        token_y: torch.Tensor,
    ):
        """
        Forward method for calculating the attention-based label smoothing loss.

        This method computes the loss using the provided dense and token labels.
        The loss is calculated only if the weight is greater than zero; otherwise,
        it returns None.

        Args:
            dense_y (torch.Tensor): The predicted dense outputs from the model.
            token_y (torch.Tensor): The ground truth token labels for the inputs.

        Returns:
            torch.Tensor or None: The computed loss value if weight > 0,
            otherwise None.

        Examples:
            >>> loss_fn = S2STAttentionLoss(vocab_size=100, smoothing=0.1)
            >>> dense_y = torch.randn(10, 100)  # Example predicted outputs
            >>> token_y = torch.randint(0, 100, (10,))  # Example token labels
            >>> loss = loss_fn(dense_y, token_y)
            >>> print(loss)  # Outputs the computed loss tensor
        """
        if self.weight > 0:
            return self.loss(dense_y, token_y)
        else:
            return None
