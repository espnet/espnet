"""E-Branchformer block for Transducer encoder."""

from typing import Dict, Optional, Tuple

import torch


class EBranchformer(torch.nn.Module):
    """E-Branchformer module definition.

    Reference: https://arxiv.org/pdf/2210.00077.pdf

    Args:
        block_size: Input/output size.
        linear_size: Linear layers' hidden size.
        self_att: Self-attention module instance.
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
        self_att: torch.nn.Module,
        feed_forward: torch.nn.Module,
        feed_forward_macaron: torch.nn.Module,
        conv_mod: torch.nn.Module,
        depthwise_conv_mod: torch.nn.Module,
        norm_class: torch.nn.Module = torch.nn.LayerNorm,
        norm_args: Dict = {},
        dropout_rate: float = 0.0,
    ) -> None:
        """Construct a E-Branchformer object."""
        super().__init__()

        self.self_att = self_att

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

        self.norm_self_att = norm_class(block_size, **norm_args)
        self.norm_feed_forward = norm_class(block_size, **norm_args)
        self.norm_feed_forward_macaron = norm_class(block_size, **norm_args)
        self.norm_mlp = norm_class(block_size, **norm_args)
        self.norm_final = norm_class(block_size, **norm_args)

        self.dropout = torch.nn.Dropout(dropout_rate)

        self.block_size = block_size
        self.linear_size = linear_size

        self.cache = None

    def reset_streaming_cache(self, left_context: int, device: torch.device) -> None:
        """Initialize/Reset self-attention and convolution modules cache for streaming.

        Args:
            left_context: Number of previous frames the attention module can see
                          in current chunk.
            device: Device to use for cache tensor.

        """
        self.cache = [
            torch.zeros(
                (1, left_context, self.block_size),
                device=device,
            ),
            torch.zeros(
                (
                    1,
                    self.linear_size // 2,
                    self.conv_mod.kernel_size - 1,
                ),
                device=device,
            ),
            torch.zeros(
                (
                    1,
                    self.block_size + self.block_size,
                    self.depthwise_conv_mod.kernel_size - 1,
                ),
                device=device,
            ),
        ]

    def forward(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor,
        mask: torch.Tensor,
        chunk_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input sequences.

        Args:
            x: E-Branchformer input sequences. (B, T, D_block)
            pos_enc: Positional embedding sequences. (B, 2 * (T - 1), D_block)
            mask: Source mask. (B, T)
            chunk_mask: Chunk mask. (T_2, T_2)

        Returns:
            x: E-Branchformer output sequences. (B, T, D_block)
            mask: Source mask. (B, T)
            pos_enc: Positional embedding sequences. (B, 2 * (T - 1), D_block)

        """
        residual = x

        x = self.norm_feed_forward_macaron(x)
        x = residual + self.feed_forward_scale * self.dropout(
            self.feed_forward_macaron(x)
        )

        x1 = x
        x2 = x

        x1 = self.norm_self_att(x1)
        x1 = self.dropout(
            self.self_att(x1, x1, x1, pos_enc, mask=mask, chunk_mask=chunk_mask)
        )

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

        return x, mask, pos_enc

    def chunk_forward(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor,
        mask: torch.Tensor,
        left_context: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode chunk of input sequence.

        Args:
            x: E-Branchformer input sequences. (B, T, D_block)
            pos_enc: Positional embedding sequences. (B, 2 * (T - 1), D_block)
            mask: Source mask. (B, T_2)
            left_context: Number of previous frames the attention module can see
                          in current chunk.

        Returns:
            x: E-Branchformer output sequences. (B, T, D_block)
            pos_enc: Positional embedding sequences. (B, 2 * (T - 1), D_block)

        """
        residual = x

        x = self.norm_feed_forward_macaron(x)
        x = residual + self.feed_forward_scale * self.feed_forward_macaron(x)

        x1 = x
        x2 = x

        x1 = self.norm_self_att(x1)

        if left_context > 0:
            key = torch.cat([self.cache[0], x1], dim=1)
        else:
            key = x1
        att_cache = key[:, -left_context:, :]

        x1 = self.self_att(x1, key, key, pos_enc, mask=mask, left_context=left_context)

        x2 = self.norm_mlp(x2)

        x2 = self.channel_proj1(x2)
        x2, conv_cache = self.conv_mod(x2, cache=self.cache[1])
        x2 = self.channel_proj2(x2)

        x_concat = torch.cat([x1, x2], dim=-1)
        x_depth, merge_cache = self.depthwise_conv_mod(x_concat, cache=self.cache[2])

        x = x + self.merge_proj(x_concat + x_depth)

        residual = x

        x = self.norm_feed_forward(x)
        x = residual + self.feed_forward_scale * self.feed_forward(x)

        x = self.norm_final(x)

        self.cache = [att_cache, conv_cache, merge_cache]

        return x, pos_enc
