"""Embedding for token IDs and differentiable token distributions."""

from typing import Optional, Tuple, Type

import torch
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.legacy.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,
)


class TokenEmbedding(AbsFrontend):
    """Embed token IDs or differentiable assignments with shared weights.

    Integer inputs of shape ``(B, T)`` use ordinary embedding lookup.  A
    floating-point input of shape ``(B, T, K)`` is interpreted as a one-hot or
    soft distribution over the ``K`` tokens and multiplied by the embedding
    matrix.  The latter path preserves gradients to a differentiable
    quantizer, including hard straight-through assignments.

    Args:
        input_size: Number of input tokens ``K``.
        embed_dim: Embedding dimension.
        use_positional_encoding: Apply positional encoding after embedding.
        pos_enc_class: Positional-encoding class.  It must accept the embedding
            dimension and dropout rate as its first two arguments.
        positional_dropout_rate: Dropout rate used by positional encoding.
        padding_idx: Optional padding index for integer token inputs.  A padded
            distribution should instead be represented by an all-zero vector.
    """

    @typechecked
    def __init__(
        self,
        input_size: int,
        embed_dim: int,
        use_positional_encoding: bool = True,
        pos_enc_class: Type[torch.nn.Module] = PositionalEncoding,
        positional_dropout_rate: float = 0.1,
        padding_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError(f"input_size must be positive: {input_size}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive: {embed_dim}")
        if not 0.0 <= positional_dropout_rate <= 1.0:
            raise ValueError(
                "positional_dropout_rate must be in [0, 1]: "
                f"{positional_dropout_rate}"
            )
        if padding_idx is not None and not -input_size <= padding_idx < input_size:
            raise ValueError(
                f"padding_idx must be in [-{input_size}, {input_size}): "
                f"{padding_idx}"
            )

        self.input_size = input_size
        self.embed_dim = embed_dim
        self.embed = torch.nn.Embedding(input_size, embed_dim, padding_idx=padding_idx)
        self.pos_enc = (
            pos_enc_class(embed_dim, positional_dropout_rate)
            if use_positional_encoding
            else torch.nn.Identity()
        )

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed token IDs or distributions without changing their lengths.

        Args:
            input: Integer token IDs of shape ``(B, T)`` or floating-point
                token distributions of shape ``(B, T, K)``.
            input_lengths: Valid lengths of shape ``(B,)``.

        Returns:
            Embedded features of shape ``(B, T, embed_dim)`` and the unchanged
            input lengths.
        """
        if input_lengths.dim() != 1:
            raise ValueError(
                "input_lengths must have shape (B,), but got "
                f"{tuple(input_lengths.shape)}"
            )
        if input.size(0) != input_lengths.size(0):
            raise ValueError(
                "Batch sizes of input and input_lengths differ: "
                f"{input.size(0)} != {input_lengths.size(0)}"
            )

        if torch.is_floating_point(input):
            if input.dim() != 3:
                raise ValueError(
                    "Floating-point input must have shape (B, T, K), but got "
                    f"{tuple(input.shape)}"
                )
            if input.size(-1) != self.input_size:
                raise ValueError(
                    "The distribution size must equal input_size: "
                    f"{input.size(-1)} != {self.input_size}"
                )
            x = torch.matmul(input.to(dtype=self.embed.weight.dtype), self.embed.weight)
        else:
            if input.dim() != 2:
                raise ValueError(
                    "Integer input must have shape (B, T), but got "
                    f"{tuple(input.shape)}"
                )
            if input.dtype == torch.bool:
                raise TypeError("Boolean token IDs are not supported")
            x = self.embed(input.to(dtype=torch.long))

        return self.pos_enc(x), input_lengths

    def output_size(self) -> int:
        """Return the embedding dimension."""
        return self.embed_dim
