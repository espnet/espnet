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
        token_per_frame: int = 1,
        pos_enc_class=PositionalEncoding,
        positional_dropout_rate: float = 0.1,
    ):
        """Initialize.

        Args:
            input_size: Number of input tokens.
            embed_dim: Embedding Size.
            token_per_frame: number of tokens per frame in the input
            pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
            positional_dropout_rate: dropout rate after adding positional encoding
        """

        super().__init__()
        self.embed_dim = embed_dim
        self.token_per_frame = token_per_frame

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
            Tensor: Output with dimensions (B, T // token_per_frame, D).
            Tensor: Output lengths within batch, devided by token_per_frame
        """

        assert input.dim() == 2, input.size()
        assert input.size(1) % self.token_per_frame == 0, input.size()
        assert torch.all(input_lengths % self.token_per_frame == 0), input_lengths

        B, T = input.size()
        x = input.view(B, T // self.token_per_frame, self.token_per_frame)
        x = self.emb(x).mean(dim=2)
        x = self.ln(self.pos(x))

        input_lengths = input_lengths // self.token_per_frame

        return x, input_lengths

    def output_size(self) -> int:
        """Return output length of feature dimension D, i.e. the embedding dim."""
        return self.embed_dim


class CodecEmbedding(AbsFrontend):
    """Use codec dequantization process and the input embeddings"""

    @typechecked
    def __init__(
        self,
        input_size,
        hf_model_tag: str = "espnet/amuse_encodec_16k",
        token_bias: int = 2,
        token_per_frame: int = 8,
        pos_enc_class=PositionalEncoding,
        positional_dropout_rate: float = 0.1,
    ):
        """Initialize.

        Args:
            hf_model_tag: HuggingFace model tag for Espnet codec models
            token_bias: the index of the first codec code
            pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
            positional_dropout_rate: dropout rate after adding positional encoding
        """

        super().__init__()

        from espnet2.bin.gan_codec_inference import AudioCoding

        model = AudioCoding.from_pretrained(model_tag=hf_model_tag).model
        self.quantizer = model.codec.generator.quantizer
        self.codebook_size = self.quantizer.bins
        self.codebook_dim = self.quantizer.codebook_dim
        self.token_bias = token_bias

        # NOTE(Jinchuan): make it as an external parameter rather than parsing from
        # the quantizer since not all codebooks will be used all the time.
        self.token_per_frame = token_per_frame

        self.vocab_size = input_size
        self.pos = pos_enc_class(self.codebook_dim, positional_dropout_rate)
        self.ln = torch.nn.LayerNorm(self.codebook_dim)

        self.decoder = model.codec.generator.decoder

    def forward(
        self,
        input: torch.Tensor,
        input_lengths: torch.Tensor,
    ):
        assert input.dim() == 2, input.size()
        assert input.size(1) % self.token_per_frame == 0, input.size()
        assert torch.all(input_lengths % self.token_per_frame == 0), input_lengths
        assert torch.all(input < self.vocab_size)

        B, Tnq = input.size()
        x = input.view(B, Tnq // self.token_per_frame, self.token_per_frame)
        x = x - self.token_bias

        for n in range(self.token_per_frame):
            x[:, :, n] -= n * self.codebook_size
        # NOTE (Jinchuan): do this clip so that the dequantization process
        # will not encounter an error. In practice, only the padding values
        # will exceed this range and is ignored by the length mask later.
        x = torch.clip(x, min=0, max=self.codebook_size - 1)

        z = self.quantizer.decode(x.permute(2, 0, 1)).permute(0, 2, 1)

        z = self.ln(z)
        z = self.pos(z)

        input_lengths = input_lengths // self.token_per_frame

        return z, input_lengths

    def output_size(self) -> int:
        """Return output length of feature dimension D, i.e. the embedding dim."""
        return self.codebook_dim
