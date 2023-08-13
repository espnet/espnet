from typing import List, Tuple

import torch
import torch.nn as nn
from typeguard import check_argument_types

from espnet2.asr.decoder.transformer_decoder import BaseTransformerDecoder
from espnet2.spk.pooling.abs_pooling import AbsPooling
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder_layer import DecoderLayer
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,
    RelPositionalEncoding,
    LearnableFourierPosEnc,
    ScaledPositionalEncoding
)
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat

class TransformerDecoderPooling(AbsPooling, BaseTransformerDecoder):
    """
    Inputs task token as an input instead of <sos>.
    Applies cross attention with the encoder output (i.e., frame-level speaker
    embeddings).
    Designed to output different embeddings from the same input utterance when
    fed with different task tokens.

    args:

    """

    def __init__(
        self,
        vocab_size: int,
        input_size: int,
        num_blocks: int = 3,
        attention_dim: int = 512,
        attention_heads: int = 4,
        linear_units: int = 2048,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        pos_enc_class="rel_pos",
        concat_after: bool = False,
        normalize_before: bool = True,
        use_output_layer: bool = False,
        layer_drop_rate: float = 0.0,
    ):
        assert check_argument_types()

        if pos_enc_class == "pos":
            pos_enc_layer = PositionalEncoding
        elif pos_enc_class == "rel_pos":
            pos_enc_layer = RelPositionalEncoding
        elif pos_enc_class == "learnable_fourier_pos":
            pos_enc_layer = LearnableFourierPosEnc
        elif pose_enc_class == "scale_pos":
            pos_enc_layer = ScaledPositionalEncoding

        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=input_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_layer,
            normalize_before=normalize_before,
        )

        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(vocab_size, attention_dim),
                pos_enc_layer(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(vocab_size, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_layer(attention_dim, positional_dropout_rate),
            )
        else:
            raise ValueError(f"only 'embed' or 'linear' is supported: {input_layer}")

        if attention_dim != input_size:
            self.encoder_mapping = nn.Linear(input_size, attention_dim)
        else:
            self.encoder_mapping = nn.Identity()
        self._output_size = attention_dim

        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

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
            layer_drop_rate,
        )

    def output_size(self):
        return self._output_size

    def forward(
        self,
        encoder_output: torch.Tensor,
        task_tokens: torch.Tensor,
    ):
        """
        Args:
            encoder_output: frame-level embeddings, (batch, dim, seq)
            task_tokens: (batch,)
        Returns:
            x: utterance-level embedding (batch, dim_out)
        """
        # (bs, seq, dim)
        memory = self.encoder_mapping(encoder_output.transpose(-2, -1))
        x = self.embed(task_tokens)

        # make masks
        x, _, memory, _ = self.decoders(x, None, memory, None)
        if self.normalize_before:
            x = self.after_norm(x)

        # (bs, dim)
        return x.squeeze(1)
