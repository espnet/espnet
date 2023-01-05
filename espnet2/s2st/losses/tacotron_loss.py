import torch
import torch.nn.functional as F

from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.e2e_tts_tacotron2 import Tacotron2Loss
from espnet2.s2st.losses.abs_loss import AbsS2STLoss
from espnet2.utils.types import str2bool


class S2STTacotron2Loss(AbsS2STLoss):
    """Tacotron-based loss for S2ST."""

    def __init__(
        self,
        weight: float = 1.0,
        loss_type: str = "L1+L2",
        use_masking: str2bool = True,
        use_weighted_masking: str2bool = False,
        bce_pos_weight: float = 20.0
    ):
        super().__init__()
        assert check_argument_types()
        self.weight = weight
        self.loss = Tacotron2Loss(
            use_masking=use_masking,
            use_weighted_masking=use_weighted_masking,
            bce_pos_weight=bce_pos_weight
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
        """Forward.

        Args:
            after_outs (Tensor): Batch of outputs after postnets (B, Lmax, odim).
            before_outs (Tensor): Batch of outputs before postnets (B, Lmax, odim).
            logits (Tensor): Batch of stop logits (B, Lmax).
            ys (Tensor): Batch of padded target features (B, Lmax, odim).
            labels (LongTensor): Batch of the sequences of stop token labels (B, Lmax).
            olens (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: L1 loss value.
            Tensor: Mean square error loss value.
            Tensor: Binary cross entropy loss value.
        """
        if self.weight > 0:
            l1_loss, mse_loss, bce_loss = self.loss(after_outs, before_outs, logits, ys, labels, olens)
            if self.loss_type == "L1+L2":
                return l1_loss + mse_loss + bce_loss, l1_loss, mse_loss, bce_loss
            elif self.loss_type == "L1":
                return l1_loss + bce_loss, l1_loss, mse_loss, bce_loss
            elif self.loss_type == "L2":
                return mse_loss + bce_loss, l1_loss, mse_loss, bce_loss
            else:
                raise ValueError(f"unknown --loss-type {self.loss_type}")
        else:
            return 0, 0, 0, 0