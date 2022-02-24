#!/usr/bin/env python3
#  2020, Technische Universität München;  Ludwig Kürzinger
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Embedding Frontend for text based inputs."""

from espnet2.asr.frontend.abs_frontend import AbsFrontend
import torch
from typeguard import check_argument_types
from typing import Tuple


class Embedding(AbsFrontend):
    """Embedding Frontend for text based inputs.
    """

    def __init__(
        self,
        input_size: int = 400,
        embed_dim: int = 400,
        no_embed_scale: bool = False,
        padding: int = -1,
    ):
        """Initialize.

        Args:
            input_size: Number of input tokens.
            embed_dim: Embedding Size.
            no_embed_scale: Whether to scale the embeddings or not.
            padding: Padding.
        """
        assert check_argument_types()
        super().__init__()
        self.embed_dim = embed_dim
        self.padding = padding
        self.embed_scale = 1.0 if no_embed_scale else math.sqrt(embed_dim)
        self.embed = torch.nn.Embedding(input_size, embed_dim, padding_idx=padding)

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply a sliding window on the input.

        Args:
            input: Input (B, T) or (B, T,D), with D.
            input_lengths: Input lengths within batch.

        Returns:
            Tensor: Output with dimensions (B, T, D).
            Tensor: Output lengths within batch.
        """
        token_embedding = self.embed(input)
        x = self.embed_scale * token_embedding

        return x, input_lengths

    def output_size(self) -> int:
        """Return output length of feature dimension D, i.e. the embedding dim."""
        return self.embed_dim
