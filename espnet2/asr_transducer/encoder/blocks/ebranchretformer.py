"""E-BranchRetformer block for Transducer encoder."""

from typing import Dict, Optional, Tuple

import torch


class EBranchRetformer(torch.nn.Module):
    """E-BranchRetformer module definition.

    References:
        https://arxiv.org/pdf/2210.00077.pdf
        https://arxiv.org/pdf/2307.08621.pdf

    Args:
        block_size: Input/output size.
        linear_size: Linear layers' hidden size.
        multi_scale_retention: MultiScaleRetention module instance.
        feed_forward: Feed-forward module instance.
        feed_forward_macaron: Feed-forward module instance for macaron network.
        conv_mod: ConvolutionalSpatialGatingUnit module instance.
        depthwise_conv_mod: DepthwiseConvolution module instance.
        norm_class: Normalization class.
        norm_args: Normalization module arguments.
        dropout_rate: Dropout rate.

    """

    def __init__(
        self,
        block_size: int,
        linear_size: int,
        multi_scale_retention: torch.nn.Module,
        feed_forward: torch.nn.Module,
        feed_forward_macaron: torch.nn.Module,
        conv_mod: torch.nn.Module,
        depthwise_conv_mod: torch.nn.Module,
        norm_class: torch.nn.Module = torch.nn.LayerNorm,
        norm_args: Dict = {},
        dropout_rate: float = 0.0,
    ) -> None:
        """Construct a E-BranchRetformer object."""
        super().__init__()

        self.multi_scale_retention = multi_scale_retention

        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.feed_forward_scale = 0.5

        self.conv_mod = conv_mod
        self.depthwise_conv_mod = depthwise_conv_mod

        self.channel_proj1 = torch.nn.Sequential(
            torch.nn.Linear(block_size, linear_size), torch.nn.GELU()
        )
        self.channel_proj2 = torch.nn.Linear(linear_size // 2, block_size)

        self.merge_proj = torch.nn.Linear((block_size + block_size), block_size)

        self.norm_retention = norm_class(block_size, **norm_args)
        self.norm_feed_forward = norm_class(block_size, **norm_args)
        self.norm_feed_forward_macaron = norm_class(block_size, **norm_args)
        self.norm_mlp = norm_class(block_size, **norm_args)
        self.norm_final = norm_class(block_size, **norm_args)

        self.dropout = torch.nn.Dropout(dropout_rate)

        self.block_size = block_size
        self.linear_size = linear_size

        self.cache = None

    def reset_streaming_cache(self, left_context: int, device: torch.device) -> None:
        """Initialize/Reset cache for streaming.

        Args:
            left_context: Number of previous frames the attention module can see
                          in current chunk.
            device: Device to use for cache tensor.

        """
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor,
        mask: torch.Tensor,
        chunk_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input sequences.

        Args:
            x: E-BranchRetformer input sequences. (B, T, D_block)

        Returns:
            x: E-BranchRetformer output sequences. (B, T, D_block)

        """
        residual = x

        x = self.norm_feed_forward_macaron(x)
        x = residual + self.feed_forward_scale * self.dropout(
            self.feed_forward_macaron(x)
        )

        x1 = x
        x2 = x

        x1 = self.norm_retention(x1)
        x1 = self.dropout(self.multi_scale_retention(x1))

        x2 = self.norm_mlp(x2)

        x2 = self.channel_proj1(x2)
        x2, _ = self.conv_mod(x2, mask=mask)
        x2 = self.dropout(self.channel_proj2(x2))

        x_concat = torch.cat([x1, x2], dim=-1)
        x_depth, _ = self.depthwise_conv_mod(x_concat, mask=mask)

        x = x + self.merge_proj(x_concat + x_depth)

        residual = x

        x = self.norm_feed_forward(x)
        x = residual + self.feed_forward_scale * self.dropout(self.feed_forward(x))

        x = self.norm_final(x)

        return x, mask, None

    def chunk_forward(
        self,
        x: torch.Tensor,
        pos_enc: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        left_context: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode chunk of input sequence.

        Args:
            x: E-BranchRetformer input sequences. (B, T, D_block)

        Returns:
            x: E-BranchRetformer output sequences. (B, T, D_block)

        """
        raise NotImplementedError
