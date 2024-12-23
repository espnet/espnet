import torch
from typeguard import typechecked

from espnet2.s2st.losses.abs_loss import AbsS2STLoss
from espnet2.utils.types import str2bool
from espnet.nets.pytorch_backend.e2e_tts_tacotron2 import Tacotron2Loss


class S2STTacotron2Loss(AbsS2STLoss):
    """
    Tacotron-based loss for S2ST.

    This class implements the loss function for the sequence-to-sequence
    text-to-speech (S2ST) model based on the Tacotron architecture. It
    incorporates various loss types including L1, L2, and Binary Cross
    Entropy (BCE) and allows for masking options during loss computation.

    Attributes:
        weight (float): Weighting factor for the loss. Default is 1.0.
        loss_type (str): Type of loss to compute. Options are "L1+L2", "L1",
            or "L2". Default is "L1+L2".
        loss (Tacotron2Loss): Instance of the Tacotron2Loss class.

    Args:
        weight (float): Weighting factor for the loss. Default is 1.0.
        loss_type (str): Type of loss to compute. Options are "L1+L2", "L1",
            or "L2". Default is "L1+L2".
        use_masking (str2bool): Flag to enable masking. Default is True.
        use_weighted_masking (str2bool): Flag to enable weighted masking.
            Default is False.
        bce_pos_weight (float): Positive weight for BCE loss. Default is 20.0.

    Returns:
        Tensor: L1 loss value.
        Tensor: Mean square error loss value.
        Tensor: Binary cross entropy loss value.

    Raises:
        ValueError: If an unknown loss type is specified.

    Examples:
        >>> loss_fn = S2STTacotron2Loss(weight=1.0, loss_type="L1")
        >>> after_outs = torch.randn(4, 100, 80)  # Example tensor
        >>> before_outs = torch.randn(4, 100, 80)  # Example tensor
        >>> logits = torch.randn(4, 100)           # Example tensor
        >>> ys = torch.randn(4, 100, 80)           # Example tensor
        >>> labels = torch.randint(0, 2, (4, 100)) # Example tensor
        >>> olens = torch.tensor([100, 90, 80, 70])  # Example tensor
        >>> loss, l1_loss, mse_loss, bce_loss = loss_fn(
        ...     after_outs, before_outs, logits, ys, labels, olens)
    """

    @typechecked
    def __init__(
        self,
        weight: float = 1.0,
        loss_type: str = "L1+L2",
        use_masking: str2bool = True,
        use_weighted_masking: str2bool = False,
        bce_pos_weight: float = 20.0,
    ):
        super().__init__()
        self.weight = weight
        self.loss_type = loss_type
        self.loss = Tacotron2Loss(
            use_masking=use_masking,
            use_weighted_masking=use_weighted_masking,
            bce_pos_weight=bce_pos_weight,
        )

    def forward(
        self,
        after_outs: torch.Tensor,
        before_outs: torch.Tensor,
        logits: torch.Tensor,
        ys: torch.Tensor,
        labels: torch.Tensor,
        olens: torch.Tensor,
    ):
        """
        Forward pass for computing the loss based on the outputs and targets.

        This method calculates the loss values for a given batch of outputs
        from the Tacotron model and the corresponding target values. The loss
        is computed based on the specified loss type (L1, L2, or a combination).

        Args:
            after_outs (Tensor): Batch of outputs after postnets (B, Lmax, odim).
            before_outs (Tensor): Batch of outputs before postnets (B, Lmax, odim).
            logits (Tensor): Batch of stop logits (B, Lmax).
            ys (Tensor): Batch of padded target features (B, Lmax, odim).
            labels (LongTensor): Batch of the sequences of stop token labels (B, Lmax).
            olens (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: Total loss value.
            Tensor: L1 loss value.
            Tensor: Mean square error loss value.
            Tensor: Binary cross entropy loss value.

        Raises:
            ValueError: If an unknown loss type is specified.

        Examples:
            >>> loss_fn = S2STTacotron2Loss()
            >>> after_outs = torch.randn(2, 10, 80)
            >>> before_outs = torch.randn(2, 10, 80)
            >>> logits = torch.randn(2, 10)
            >>> ys = torch.randn(2, 10, 80)
            >>> labels = torch.tensor([[1, 0, 1, 1, 0, 0, 1, 0, 0, 0],
            ...                        [0, 1, 0, 0, 1, 1, 0, 0, 0, 0]])
            >>> olens = torch.tensor([10, 10])
            >>> total_loss, l1_loss, mse_loss, bce_loss = loss_fn(
            ...     after_outs, before_outs, logits, ys, labels, olens)
        """
        if self.weight > 0:
            l1_loss, mse_loss, bce_loss = self.loss(
                after_outs, before_outs, logits, ys, labels, olens
            )
            if self.loss_type == "L1+L2":
                return l1_loss + mse_loss + bce_loss, l1_loss, mse_loss, bce_loss
            elif self.loss_type == "L1":
                return l1_loss + bce_loss, l1_loss, mse_loss, bce_loss
            elif self.loss_type == "L2":
                return mse_loss + bce_loss, l1_loss, mse_loss, bce_loss
            else:
                raise ValueError(f"unknown --loss-type {self.loss_type}")
        else:
            return None, None, None, None
