"""Receptance Weighted Key Value (RWKV) block definition.

Based/modified from https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4/src/model.py

"""

from typing import Dict, Optional, Tuple

import torch

from espnet2.asr_transducer.decoder.modules.rwkv.attention import SelfAttention
from espnet2.asr_transducer.decoder.modules.rwkv.feed_forward import FeedForward


class RWKV(torch.nn.Module):
    """RWKV module.

    Args:
        size: Input/Output size.
        linear_size: Feed-forward hidden size.
        attention_size: SelfAttention hidden size.
        context_size: Context size for WKV computation.
        block_id: Block index.
        num_blocks: Number of blocks in the architecture.
        normalization_class: Normalization layer class.
        normalization_args: Normalization layer arguments.
        att_dropout_rate: Dropout rate for the attention module.
        ffn_dropout_rate: Dropout rate for the feed-forward module.

    """

    def __init__(
        self,
        size: int,
        linear_size: int,
        attention_size: int,
        context_size: int,
        block_id: int,
        num_blocks: int,
        normalization_class: torch.nn.Module = torch.nn.LayerNorm,
        normalization_args: Dict = {},
        att_dropout_rate: float = 0.0,
        ffn_dropout_rate: float = 0.0,
    ) -> None:
        """Construct a RWKV object."""
        super().__init__()

        self.layer_norm_att = normalization_class(size, **normalization_args)
        self.layer_norm_ffn = normalization_class(size, **normalization_args)

        self.att = SelfAttention(
            size, attention_size, context_size, block_id, num_blocks
        )
        self.dropout_att = torch.nn.Dropout(p=att_dropout_rate)

        self.ffn = FeedForward(size, linear_size, block_id, num_blocks)
        self.dropout_ffn = torch.nn.Dropout(p=ffn_dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute receptance weighted key value.

        Args:
            x: RWKV input sequences. (B, L, size)
            state: Decoder hidden states. [5 x (B, D_att/size, N)]

        Returns:
            x: RWKV output sequences. (B, L, size)
            x: Decoder hidden states. [5 x (B, D_att/size, N)]

        """
        att, state = self.att(self.layer_norm_att(x), state=state)
        x = x + self.dropout_att(att)

        ffn, state = self.ffn(self.layer_norm_ffn(x), state=state)
        x = x + self.dropout_ffn(ffn)

        return x, state
