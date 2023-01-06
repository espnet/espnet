import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet2.s2st.losses.abs_loss import AbsS2STLoss


class S2STCTCLoss(AbsS2STLoss):
    """CTC-based loss for S2ST."""

    def __init__(
        self,
        weight: float = 1.0,
    ):
        # Note(Jiatong): dummy CTC loss, only providing weight
        # for final loss calculation
        super().__init__()
        assert check_argument_types()
        self.weight = weight
