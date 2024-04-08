import torch
from typeguard import typechecked

from espnet2.s2st.losses.abs_loss import AbsS2STLoss
from espnet2.utils.types import str2bool
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (  # noqa: H301
    LabelSmoothingLoss,
)


class S2STAttentionLoss(AbsS2STLoss):
    """attention-based label smoothing loss for S2ST."""

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
        """Forward.

        Args:
        """
        if self.weight > 0:
            return self.loss(dense_y, token_y)
        else:
            return None
