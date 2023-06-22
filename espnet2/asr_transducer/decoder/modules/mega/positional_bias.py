"""Positional bias related modules.

Based/modified from https://github.com/facebookresearch/mega/blob/main/fairseq/modules/relative_positional_bias.py
"""  # noqa

import math
from typing import Tuple

import torch


class RelativePositionBias(torch.nn.Module):
    """RelativePositionBias module definition.

    Args:
        max_positions: Maximum number of relative positions.

    """

    def __init__(self, max_positions: int) -> None:
        """Construct a RelativePositionBias object."""
        super().__init__()

        self.max_positions = max_positions

        self.relative_position_bias = torch.nn.Parameter(
            torch.Tensor(2 * self.max_positions - 1)
        )

        self.reset_parameters()

    def reset_parameters(self, val: float = 0.0, std: float = 0.02) -> None:
        """Reset module parameters.

        Args:
            val: Initialization value.
            std: Standard deviation.

        """
        torch.nn.init.normal_(self.relative_position_bias, mean=val, std=std)

    def forward(self, length: int) -> torch.Tensor:
        """Compute relative position bias.

        Args:
            length: Sequence length.

        Returns:
            tile: Relative position bias. (L, L)

        """
        if length > self.max_positions:
            raise ValueError(
                f"Length {length} is too long for the maximum number of "
                f"allowed positions {self.max_positions}."
            )

        bias = self.relative_position_bias[
            (self.max_positions - length) : (self.max_positions + length - 1)
        ]
        bias = torch.nn.functional.pad(bias, (0, length))

        tile = torch.tile(bias, (length,))[:-length]
        tile = tile.view(length, (3 * length - 2))

        start = (2 * length - 1) // 2
        end = tile.size(1) - start

        tile = tile[:, start:end]

        return tile


class RotaryRelativePositionBias(torch.nn.Module):
    """RotaryRelativePositionBias module definition.

    Args:
        size: Module embedding size.
        max_positions: Maximum number of relative positions.

    """

    def __init__(self, size: int, max_positions: int = 2048) -> None:
        """Construct a RotaryRelativePositionBias object."""
        super().__init__()

        self.sine, self.cosine = RotaryRelativePositionBias.get_sinusoid_embeddings(
            max_positions, size
        )

        self.alpha = torch.nn.Parameter(torch.Tensor(1, size))
        self.beta = torch.nn.Parameter(torch.Tensor(1, size))

        self.register_buffer("_pe", torch.FloatTensor(1))

        self.size = size
        self.max_positions = max_positions

        self.reset_parameters()

    def reset_parameters(self, val: float = 0.0, std: float = 0.02) -> None:
        """Reset module parameters.

        Args:
            val: Initialization value.
            std: Standard deviation.

        """
        torch.nn.init.normal_(self.alpha, mean=val, std=std)
        torch.nn.init.normal_(self.beta, mean=val, std=std)

    @staticmethod
    def get_sinusoid_embeddings(
        max_positions: int,
        size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute sinusoidal positional embeddings.

        Args:
            max_positions: Maximum number of positions.
            size: Input size.

        Returns:
            : Sine elements. (max_positions, size // 2)
            : Cos elements. (max_positions, size // 2)

        """
        half_size = size // 2

        emb = math.log(10000) / half_size
        emb = torch.exp(torch.arange(half_size, dtype=torch.float) * -emb)
        emb = torch.arange(max_positions, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)

        return torch.sin(emb), torch.cos(emb)

    def rotary(self, x: torch.Tensor) -> torch.Tensor:
        """Compute rotary positional embeddings.

        Args:
            x: Input sequence. (L, size)

        Returns:
            x: Rotary positional embeddings.  (L, size)

        """
        length, dim = x.size()

        x1, x2 = torch.chunk(x, 2, dim=-1)

        if self.sine is None or length > self.sine.size(0):
            self.sine, self.cosine = RotaryRelativePositionBias.get_sinusoid_embeddings(
                length, dim
            )

            self.max_positions = length

        self.sine = self.sine.to(self._pe)
        self.cosine = self.cosine.to(self._pe)

        sin = self.sine[:length]
        cos = self.cosine[:length]

        x = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=1)

        return x

    def forward(self, length: int) -> torch.Tensor:
        """Compute rotary relative position bias.

        Args:
            length: Sequence length.

        Returns:
            bias: Rotary relative position bias. (L, L)

        """
        alpha = self.rotary(self.alpha.expand(length, self.size))
        beta = self.rotary(self.beta.expand(length, self.size))

        bias = torch.einsum("mk, nk -> mn", alpha, beta)

        return bias
