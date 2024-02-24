"""XPOS-style rotary positional embedding module for retention."""

from typing import Optional, Tuple

import torch


class XPOSRetentionPositionalEmbedding(torch.nn.Module):
    """XPOSRetentionPositionalEmbedding module definition.

    Args:
        size: Embedding size.
        num_heads: Number of retention heads.

    """

    def __init__(self, size: int, num_heads: int) -> None:
        """Construct a XPOSRetentionPositionalEmbedding object."""
        super().__init__()

        angle = 1.0 / (10000 ** torch.linspace(0, 1, size // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()

        decay = torch.log(1 - 2 ** (-5 - torch.arange(num_heads, dtype=torch.float)))

        self.size = size

        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)

    def forward(
        self, x: torch.Tensor, recurrent: bool = False, left_context: int = 0
    ) -> torch.Tensor:
        """Compute frequencies and decaying mask for given length.

        Args:
            length: XPOSRetentionPositionalEmbedding input sequence. (length, size)
            recurrent: Whether chunk-wise recurrent retention is performed.

        Returns:
            cos: Cosine functions. (length, D_angle) or (D_angle)
            sin: Sin functions. (length, D_angle) or (D_angle)
            : Decaying mask. (num_heads, length, length) or (num_heads)

        """
        length = x.size(1)

        if recurrent:
            raise NotImplementedError
        else:
            index = torch.arange(length).to(self.decay)

            cos = torch.cos(index[:, None] * self.angle[None, :])
            sin = torch.sin(index[:, None] * self.angle[None, :])

            mask = torch.tril(torch.ones(length, length).to(self.decay))
            mask = torch.masked_fill(
                index[:, None] - index[None, :], ~mask.bool(), float("inf")
            )
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)

            mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()

            pos = ((cos, sin), mask)

        return pos
