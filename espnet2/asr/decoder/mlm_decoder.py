# Copyright 2022 Yosuke Higuchi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Masked LM Decoder definition."""
from typing import Tuple

import torch
from typeguard import typechecked

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder_layer import DecoderLayer
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat


class MLMDecoder(AbsDecoder):
    """
        Masked Language Model (MLM) Decoder for sequence-to-sequence models.

    This class implements a decoder for Masked Language Modeling tasks, typically
    used in transformer-based architectures. It processes encoded input sequences
    and generates output sequences with the ability to handle masked tokens.

    Attributes:
        embed (torch.nn.Sequential): Input embedding layer.
        normalize_before (bool): Whether to apply layer normalization before each decoder block.
        after_norm (LayerNorm): Layer normalization applied after all decoder blocks if normalize_before is True.
        output_layer (torch.nn.Linear or None): Linear layer for final output projection.
        decoders (torch.nn.ModuleList): List of decoder layers.

    Args:
        vocab_size (int): Size of the vocabulary.
        encoder_output_size (int): Dimensionality of the encoder output.
        attention_heads (int, optional): Number of attention heads. Defaults to 4.
        linear_units (int, optional): Number of units in position-wise feed-forward layers. Defaults to 2048.
        num_blocks (int, optional): Number of decoder layers. Defaults to 6.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
        positional_dropout_rate (float, optional): Dropout rate for positional encoding. Defaults to 0.1.
        self_attention_dropout_rate (float, optional): Dropout rate for self-attention. Defaults to 0.0.
        src_attention_dropout_rate (float, optional): Dropout rate for source attention. Defaults to 0.0.
        input_layer (str, optional): Type of input layer, either "embed" or "linear". Defaults to "embed".
        use_output_layer (bool, optional): Whether to use an output layer. Defaults to True.
        pos_enc_class (class, optional): Positional encoding class. Defaults to PositionalEncoding.
        normalize_before (bool, optional): Whether to apply layer normalization before each block. Defaults to True.
        concat_after (bool, optional): Whether to concat attention layer's input and output. Defaults to False.

    Note:
        This decoder adds an extra token to the vocabulary size to account for the mask token.

    Example:
        >>> encoder_output_size = 256
        >>> vocab_size = 1000
        >>> decoder = MLMDecoder(vocab_size, encoder_output_size)
        >>> hs_pad = torch.randn(32, 50, encoder_output_size)  # (batch, max_len, feat)
        >>> hlens = torch.full((32,), 50, dtype=torch.long)
        >>> ys_in_pad = torch.randint(0, vocab_size, (32, 30))  # (batch, max_len)
        >>> ys_in_lens = torch.full((32,), 30, dtype=torch.long)
        >>> decoded, olens = decoder(hs_pad, hlens, ys_in_pad, ys_in_lens)
    """

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
        super().__init__()
        attention_dim = encoder_output_size
        vocab_size += 1  # for mask token

        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(vocab_size, attention_dim),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(vocab_size, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        else:
            raise ValueError(f"only 'embed' or 'linear' is supported: {input_layer}")

        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
        if use_output_layer:
            self.output_layer = torch.nn.Linear(attention_dim, vocab_size)
        else:
            self.output_layer = None

        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads, attention_dim, self_attention_dropout_rate
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
                Forward pass of the MLM Decoder.

        This method processes the input sequences through the decoder layers,
        applying self-attention, source attention, and feed-forward operations.

        Args:
            hs_pad (torch.Tensor): Encoded memory, float32 tensor of shape (batch, maxlen_in, feat).
            hlens (torch.Tensor): Lengths of encoded sequences, shape (batch,).
            ys_in_pad (torch.Tensor): Input token ids, int64 tensor of shape (batch, maxlen_out)
                if input_layer is "embed", or input tensor of shape (batch, maxlen_out, #mels)
                for other input layer types.
            ys_in_lens (torch.Tensor): Lengths of input sequences, shape (batch,).

        Returns:
            tuple: A tuple containing:
                - x (torch.Tensor): Decoded token scores before softmax, shape (batch, maxlen_out, token)
                  if use_output_layer is True. If use_output_layer is False, returns the
                  last decoder layer's output.
                - olens (torch.Tensor): Output sequence lengths, shape (batch,).

        Raises:
            ValueError: If the input shapes are inconsistent with the expected dimensions.

        Example:
            >>> decoder = MLMDecoder(1000, 256)
            >>> hs_pad = torch.randn(32, 50, 256)
            >>> hlens = torch.full((32,), 50, dtype=torch.long)
            >>> ys_in_pad = torch.randint(0, 1000, (32, 30))
            >>> ys_in_lens = torch.full((32,), 30, dtype=torch.long)
            >>> decoded, olens = decoder.forward(hs_pad, hlens, ys_in_pad, ys_in_lens)
            >>> print(decoded.shape, olens.shape)
            torch.Size([32, 30, 1001]) torch.Size([32])

        Note:
            The method handles masking for both the target and memory sequences to ensure
            proper attention mechanisms during decoding.
        """
        tgt = ys_in_pad
        # tgt_mask: (B, 1, L)
        tgt_mask = (~make_pad_mask(ys_in_lens)[:, None, :]).to(tgt.device)
        tgt_max_len = tgt_mask.size(-1)
        # tgt_mask_tmp: (B, L, L)
        tgt_mask_tmp = tgt_mask.transpose(1, 2).repeat(1, 1, tgt_max_len)
        tgt_mask = tgt_mask.repeat(1, tgt_max_len, 1) & tgt_mask_tmp

        memory = hs_pad
        memory_mask = (~make_pad_mask(hlens))[:, None, :].to(memory.device)

        x = self.embed(tgt)
        x, tgt_mask, memory, memory_mask = self.decoders(
            x, tgt_mask, memory, memory_mask
        )
        if self.normalize_before:
            x = self.after_norm(x)
        if self.output_layer is not None:
            x = self.output_layer(x)

        olens = tgt_mask.sum(1)
        return x, olens
