# Copyright 2022 Yosuke Higuchi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Masked LM Decoder definition."""

from typing import Any, List, Sequence, Tuple

import torch
from typeguard import typechecked

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.legacy.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.legacy.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,
)
from espnet2.legacy.nets.pytorch_backend.transformer.decoder_layer import DecoderLayer
from espnet2.legacy.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet2.legacy.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet2.legacy.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet2.legacy.nets.pytorch_backend.transformer.repeat import repeat


class MLMDecoder(AbsDecoder):
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
        layer_drop_rate: float = 0.0,
        qk_norm: bool = False,
        use_flash_attn: bool = True,
        gradient_checkpoint_layers: List[int] = [],
    ):
        super().__init__()
        attention_dim = encoder_output_size
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
        self.use_flash_attn = use_flash_attn
        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads,
                    attention_dim,
                    self_attention_dropout_rate,
                    False,  # qk_norm
                    use_flash_attn,
                    False,  # causal
                    False,  # cross_attn
                ),
                MultiHeadedAttention(
                    attention_heads,
                    attention_dim,
                    src_attention_dropout_rate,
                    False,  # qk_norm
                    use_flash_attn,
                    False,  # causal
                    True,  # cross_attn
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
        """Forward decoder.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            ys_in_pad:
                input token ids, int64 (batch, maxlen_out)
                if input_layer == "embed"
                input tensor (batch, maxlen_out, #mels) in the other cases
            ys_in_lens: (batch)
        Returns:
            (tuple): tuple containing:
            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """
        # breakpoint()
        tgt = ys_in_pad
        memory = hs_pad
        tgt_mask = None
        # Create masks
        if memory is None or memory.size(0) == 1:
            memory_mask = None
        else:
            # tgt_mask: (B, 1, L)
            # tgt_mask = (~make_pad_mask(ys_in_lens)[:, None, :]).to(tgt.device)
            # tgt_max_len = tgt_mask.size(-1)
            # # tgt_mask_tmp: (B, L, L)
            # tgt_mask_tmp = tgt_mask.transpose(1, 2).repeat(1, 1, tgt_max_len)
            # tgt_mask = tgt_mask.repeat(1, tgt_max_len, 1) & tgt_mask_tmp
            memory_mask = (~make_pad_mask(hlens))[:, None, :]  # .to(memory.device)
        # breakpoint()

        x = self.embed(tgt)
        x, tgt_mask, memory, memory_mask = self.decoders(
            x, tgt_mask, memory, memory_mask
        )

        if self.normalize_before:
            x = self.after_norm(x)
        if self.output_layer is not None:
            x = self.output_layer(x)

        olens = (
            tgt_mask.sum(1)
            if tgt_mask is not None
            else torch.LongTensor([x.size(1)]).to(x.device)
        )
        return x, olens
