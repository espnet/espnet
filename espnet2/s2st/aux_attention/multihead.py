import torch
from typeguard import typechecked

from espnet2.s2st.aux_attention.abs_aux_attention import AbsS2STAuxAttention
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention


class MultiHeadAttention(AbsS2STAuxAttention):
    """Multihead Attention for S2ST."""

    @typechecked
    def __init__(
        self,
        n_head: int = 4,
        n_feat: int = 512,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.attn = MultiHeadedAttention(
            n_head=n_head,
            n_feat=n_feat,
            dropout_rate=dropout_rate,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ):
        """Forward.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
        """
        return self.attn(query, key, value, mask)
