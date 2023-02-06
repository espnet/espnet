import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet2.s2st.losses.abs_loss import AbsS2STLoss
from espnet2.utils.types import str2bool
from espnet.nets.pytorch_backend.e2e_tts_tacotron2 import GuidedAttentionLoss
from espnet.nets.pytorch_backend.nets_utils import to_device


class S2STGuidedAttentionLoss(AbsS2STLoss):
    """Tacotron-based loss for S2ST."""

    def __init__(
        self,
        weight: float = 1.0,
        sigma: float = 0.4,
        alpha: float = 1.0,
    ):
        super().__init__()
        assert check_argument_types()
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
        """Forward.

        Args:

        Returns:
            Tensor: guided attention loss
        """
        if self.weight > 0:
            return self.loss(att_ws, ilens, olens_in)
        else:
            return None, None, None, None
