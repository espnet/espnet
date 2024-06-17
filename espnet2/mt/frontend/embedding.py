#!/usr/bin/env python3
#  2020, Technische Universität München;  Ludwig Kürzinger
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Embedding Frontend for text based inputs."""

from typing import Tuple

import torch
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding


class Embedding(AbsFrontend):
    """Embedding Frontend for text based inputs."""

    @typechecked
    def __init__(
        self,
        input_size: int = 400,
        embed_dim: int = 400,
        pos_enc_class=PositionalEncoding,
        positional_dropout_rate: float = 0.1,
    ):
        """Initialize.

        Args:
            input_size: Number of input tokens.
            embed_dim: Embedding Size.
            pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
            positional_dropout_rate: dropout rate after adding positional encoding
        """
        super().__init__()
        self.embed_dim = embed_dim
        # TODO(sdalmia): check for padding idx
        self.embed = torch.nn.Sequential(
            torch.nn.Embedding(input_size, embed_dim),
            pos_enc_class(embed_dim, positional_dropout_rate),
        )

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
        x = self.embed(input)

        return x, input_lengths

    def output_size(self) -> int:
        """Return output length of feature dimension D, i.e. the embedding dim."""
        return self.embed_dim


class PatchEmbedding(AbsFrontend):
    """Embedding Frontend for text based inputs."""

    @typechecked
    def __init__(
        self,
        input_size: int = 400,
        embed_dim: int = 400,
        patch_size: int = 1,
        pos_enc_class=PositionalEncoding,
        positional_dropout_rate: float = 0.1,
    ):
        """Initialize.

        Args:
            input_size: Number of input tokens.
            embed_dim: Embedding Size.
            patch_size: number of token per patch to sum up the embeddings
            pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
            positional_dropout_rate: dropout rate after adding positional encoding
        """

        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.emb = torch.nn.Embedding(input_size, embed_dim)
        self.pos = pos_enc_class(embed_dim, positional_dropout_rate)
        self.ln = torch.nn.LayerNorm(embed_dim)

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply a sliding window on the input.

        Args:
            input: Input (B, T)
            input_lengths: Input lengths within batch.

        Returns:
            Tensor: Output with dimensions (B, T // patch_size, D).
            Tensor: Output lengths within batch, devided by patch_size
        """

        assert input.dim() == 2, input.size()
        assert input.size(1) % self.patch_size == 0, input.size()
        assert torch.all(input_lengths % self.patch_size == 0), input_lengths

        B, T = input.size()
        x = input.view(B, T // self.patch_size, self.patch_size)
        x = self.emb(x).mean(dim=2)
        x = self.ln(self.pos(x))

        input_lengths = input_lengths // self.patch_size

        return x, input_lengths

    def output_size(self) -> int:
        """Return output length of feature dimension D, i.e. the embedding dim."""
        return self.embed_dim
