"""Multi-scale retention module."""

import math
from typing import Dict, Optional, Tuple

import torch

from espnet2.asr_transducer.encoder.modules.retention import Retention


class MultiScaleRetention(torch.nn.Module):
    """MultiScaleRetention module definition.

    Args:
        num_heads: Number of attention heads.
        size: Hidden size.
        activation: Activation module.
        dropout_rate: Dropout rate for the retention module.

    """

    def __init__(
        self,
        size: int,
        num_heads: int,
        activation: torch.nn.Module,
        decay_length: int = 768,
        dropout_rate: float = 0.0,
    ) -> None:
        """Construct a MultiScaleRetention object."""
        super().__init__()

        head_size = size // num_heads

        self.W_G = torch.nn.Parameter(torch.randn((size, size)) / size)
        self.W_O = torch.nn.Parameter(torch.randn((size, size)) / size)

        self.retention = Retention(
            size, num_heads, decay_length=decay_length, dropout_rate=dropout_rate
        )

        self.activation = activation
        self.norm = torch.nn.GroupNorm(num_heads, size)

        self.size = size
        self.head_size = head_size
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform multi-scale parallel retention.

        Args:
            x: MultiScaleRetention input sequences. (B, T, D_enc)

        Returns:
            x_out: MultiScaleRetention output sequences. (B, T, D_enc)

        """
        x_out = self.retention(x)
        x_out = self.norm(x.reshape(-1, self.size)).reshape(x.shape)

        x_out = self.activation(x @ self.W_G + x_out) @ self.W_O

        return x_out

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """Perform multi-scale recurrent retention.

        Args:
            x: MultiScaleRetention input sequences. (B, 1, D_enc)

        Returns:
            x_out: MultiScaleRetention output sequences. (B, 1, D_enc)

        """
        raise NotImplementedError
