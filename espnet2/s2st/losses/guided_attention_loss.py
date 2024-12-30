import torch
from typeguard import typechecked

from espnet2.s2st.losses.abs_loss import AbsS2STLoss
from espnet.nets.pytorch_backend.e2e_tts_tacotron2 import GuidedAttentionLoss


class S2STGuidedAttentionLoss(AbsS2STLoss):
    """
    Tacotron-based loss for S2ST.

    This class implements a guided attention loss for sequence-to-sequence
    translation tasks using the Tacotron architecture. It leverages the
    GuidedAttentionLoss from the ESPnet library to compute the loss based
    on attention weights, input lengths, and output lengths.

    Attributes:
        weight (float): Weight for the loss. If set to 0, the loss will not
            be computed.
        loss (GuidedAttentionLoss): Instance of GuidedAttentionLoss used to
            compute the guided attention loss.

    Args:
        weight (float, optional): The weight for the loss. Defaults to 1.0.
        sigma (float, optional): The sigma parameter for the guided attention
            loss. Defaults to 0.4.
        alpha (float, optional): The alpha parameter for the guided attention
            loss. Defaults to 1.0.

    Returns:
        Tensor: The guided attention loss, or None if weight is 0.

    Examples:
        >>> loss_fn = S2STGuidedAttentionLoss(weight=1.0, sigma=0.4, alpha=1.0)
        >>> att_ws = torch.rand(10, 20)  # Example attention weights
        >>> ilens = torch.randint(1, 20, (10,))  # Example input lengths
        >>> olens_in = torch.randint(1, 20, (10,))  # Example output lengths
        >>> loss = loss_fn(att_ws, ilens, olens_in)
        >>> print(loss)  # Should output the computed loss tensor

    Note:
        Ensure that the input tensors (att_ws, ilens, olens_in) are properly
        formatted and match the expected dimensions.

    Raises:
        ValueError: If the input tensors do not have compatible dimensions.
    """

    @typechecked
    def __init__(
        self,
        weight: float = 1.0,
        sigma: float = 0.4,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.weight = weight
        self.loss = GuidedAttentionLoss(
            sigma=sigma,
            alpha=alpha,
        )

    def forward(
        self,
        att_ws: torch.Tensor,
        ilens: torch.Tensor,
        olens_in: torch.Tensor,
    ):
        """
            Forward pass for calculating the guided attention loss.

        This method computes the guided attention loss based on the provided
        attention weights, input lengths, and output lengths. If the weight
        for the loss is greater than zero, it will invoke the loss calculation
        from the GuidedAttentionLoss class.

        Args:
            att_ws (torch.Tensor): Attention weights of the model output.
            ilens (torch.Tensor): Input lengths for the batch.
            olens_in (torch.Tensor): Output lengths for the batch.

        Returns:
            Tensor: Guided attention loss if weight > 0, otherwise returns
            (None, None, None, None).

        Examples:
            >>> att_ws = torch.rand(10, 50)  # Example attention weights
            >>> ilens = torch.tensor([50] * 10)  # Example input lengths
            >>> olens_in = torch.tensor([40] * 10)  # Example output lengths
            >>> loss = S2STGuidedAttentionLoss(weight=1.0)
            >>> result = loss.forward(att_ws, ilens, olens_in)
            >>> print(result)  # Should output the computed guided attention loss
        """
        if self.weight > 0:
            return self.loss(att_ws, ilens, olens_in)
        else:
            return None, None, None, None
