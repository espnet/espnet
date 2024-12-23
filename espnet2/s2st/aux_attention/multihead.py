import torch
from typeguard import typechecked

from espnet2.s2st.aux_attention.abs_aux_attention import AbsS2STAuxAttention
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention


class MultiHeadAttention(AbsS2STAuxAttention):
    """
        Multihead Attention for S2ST.

    This class implements the Multi-Head Attention mechanism specifically designed
    for sequence-to-sequence tasks. It leverages the MultiHeadedAttention
    class from the PyTorch backend Transformer implementation.

    Attributes:
        attn (MultiHeadedAttention): An instance of MultiHeadedAttention that
        performs the attention computation.

    Args:
        n_head (int): Number of attention heads. Default is 4.
        n_feat (int): Dimensionality of the feature vectors. Default is 512.
        dropout_rate (float): Dropout rate for the attention weights. Default is 0.0.

    Examples:
        >>> multihead_attn = MultiHeadAttention(n_head=8, n_feat=256)
        >>> query = torch.rand(32, 10, 256)  # Batch of 32, 10 time steps
        >>> key = torch.rand(32, 15, 256)    # Batch of 32, 15 time steps
        >>> value = torch.rand(32, 15, 256)  # Batch of 32, 15 time steps
        >>> mask = torch.ones(32, 1, 15)      # Mask for the keys
        >>> output = multihead_attn(query, key, value, mask)

    Returns:
        torch.Tensor: Output tensor with shape (#batch, time1, d_model), where
        d_model is the dimensionality of the output features.

    Raises:
        ValueError: If the dimensions of the input tensors do not match the
        expected shapes.
    """

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
        """
            Forward method for the MultiHeadAttention class.

        This method computes the attention scores using the provided query, key,
        and value tensors along with an optional mask. It utilizes the internal
        MultiHeadedAttention mechanism to perform the attention computation.

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
