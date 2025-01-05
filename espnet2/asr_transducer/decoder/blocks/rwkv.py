"""Receptance Weighted Key Value (RWKV) block definition.

Based/modified from https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4/src/model.py

"""

from typing import Dict, Optional, Tuple

import torch

from espnet2.asr_transducer.decoder.modules.rwkv.attention import SelfAttention
from espnet2.asr_transducer.decoder.modules.rwkv.feed_forward import FeedForward


class RWKV(torch.nn.Module):
    """
    Receptance Weighted Key Value (RWKV) block definition.

    This module implements the RWKV architecture, which combines recurrent and
    transformer-like properties for processing sequences. It is based on the work
    by BlinkDL, as referenced in the repository:
    https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4/src/model.py

    Attributes:
        layer_norm_att: Layer normalization for the attention module.
        layer_norm_ffn: Layer normalization for the feed-forward module.
        att: Self-attention mechanism.
        dropout_att: Dropout layer for the attention module.
        ffn: Feed-forward network.
        dropout_ffn: Dropout layer for the feed-forward network.

    Args:
        size (int): Input/Output size.
        linear_size (int): Feed-forward hidden size.
        attention_size (int): SelfAttention hidden size.
        context_size (int): Context size for WKV computation.
        block_id (int): Block index.
        num_blocks (int): Number of blocks in the architecture.
        normalization_class (torch.nn.Module, optional): Normalization layer class.
            Defaults to torch.nn.LayerNorm.
        normalization_args (Dict, optional): Normalization layer arguments.
            Defaults to an empty dictionary.
        att_dropout_rate (float, optional): Dropout rate for the attention module.
            Defaults to 0.0.
        ffn_dropout_rate (float, optional): Dropout rate for the feed-forward
            module. Defaults to 0.0.

    Examples:
        >>> rwkv = RWKV(size=512, linear_size=2048, attention_size=256,
        ...             context_size=128, block_id=0, num_blocks=12)
        >>> input_tensor = torch.randn(32, 10, 512)  # (Batch, Length, Size)
        >>> output, state = rwkv(input_tensor)

    Note:
        The RWKV architecture is designed to handle long-range dependencies
        in sequence data, leveraging the advantages of both recurrent and
        attention-based models.

    Raises:
        ValueError: If any of the input parameters are invalid.
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
        """
        Receptance Weighted Key Value (RWKV) block definition.

        Based/modified from https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4/src/model.py

        Attributes:
            layer_norm_att: Normalization layer for the attention module.
            layer_norm_ffn: Normalization layer for the feed-forward module.
            att: SelfAttention module for computing attention.
            dropout_att: Dropout layer for the attention output.
            ffn: FeedForward module for processing inputs.
            dropout_ffn: Dropout layer for the feed-forward output.

        Args:
            size (int): Input/Output size.
            linear_size (int): Feed-forward hidden size.
            attention_size (int): SelfAttention hidden size.
            context_size (int): Context size for WKV computation.
            block_id (int): Block index.
            num_blocks (int): Number of blocks in the architecture.
            normalization_class (torch.nn.Module, optional): Normalization layer class.
                Defaults to torch.nn.LayerNorm.
            normalization_args (Dict, optional): Normalization layer arguments.
                Defaults to {}.
            att_dropout_rate (float, optional): Dropout rate for the attention module.
                Defaults to 0.0.
            ffn_dropout_rate (float, optional): Dropout rate for the feed-forward module.
                Defaults to 0.0.

        Methods:
            forward(x: torch.Tensor, state: Optional[torch.Tensor]) ->
                Tuple[torch.Tensor, Optional[torch.Tensor]]:
                Compute receptance weighted key value.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - x: RWKV output sequences. Shape: (B, L, size).
                - state: Decoder hidden states. Shape: [5 x (B, D_att/size, N)].

        Examples:
            >>> rwkv = RWKV(size=256, linear_size=512, attention_size=128,
            ...            context_size=32, block_id=0, num_blocks=1)
            >>> input_tensor = torch.randn(10, 20, 256)  # (B, L, size)
            >>> output, hidden_state = rwkv(input_tensor)

        Note:
            The RWKV module is designed to work within the context of
            Receptance Weighted Key Value computations for neural network models.
        """
        att, state = self.att(self.layer_norm_att(x), state=state)
        x = x + self.dropout_att(att)

        ffn, state = self.ffn(self.layer_norm_ffn(x), state=state)
        x = x + self.dropout_ffn(ffn)

        return x, state
