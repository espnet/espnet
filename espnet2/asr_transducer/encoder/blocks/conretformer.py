"""ConRetformer block for Transducer encoder."""

from typing import Dict, Optional, Tuple

import torch


class ConRetformer(torch.nn.Module):
    """ConRetformer module definition.

    References:
        https://arxiv.org/pdf/2005.08100.pdf
        https://arxiv.org/pdf/2307.08621.pdf

    Args:
        block_size: Input/output size.
        multi_scale_retention: MultiScaleRetention module instance.
        feed_forward: Feed-forward module instance.
        feed_forward_macaron: Feed-forward module instance for macaron network.
        conv_mod: Convolution module instance.
        norm_class: Normalization module class.
        norm_args: Normalization module arguments.
        dropout_rate: Dropout rate.

    """

    def __init__(
        self,
        block_size: int,
        multi_scale_retention: torch.nn.Module,
        feed_forward: torch.nn.Module,
        feed_forward_macaron: torch.nn.Module,
        conv_mod: torch.nn.Module,
        norm_class: torch.nn.Module = torch.nn.LayerNorm,
        norm_args: Dict = {},
        dropout_rate: float = 0.0,
    ) -> None:
        """Construct a ConRetformer object."""
        super().__init__()

        self.multi_scale_retention = multi_scale_retention

        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.feed_forward_scale = 0.5

        self.conv_mod = conv_mod

        self.norm_feed_forward = norm_class(block_size, **norm_args)
        self.norm_retention = norm_class(block_size, **norm_args)

        self.norm_macaron = norm_class(block_size, **norm_args)
        self.norm_conv = norm_class(block_size, **norm_args)
        self.norm_final = norm_class(block_size, **norm_args)

        self.dropout = torch.nn.Dropout(dropout_rate)

        self.block_size = block_size
        self.cache = None

    def reset_streaming_cache(self, left_context: int, device: torch.device) -> None:
        """Initialize/Reset self-attention and convolution modules cache for streaming.

        Args:
            left_context: Number of previous frames the attention module can see
                          in current chunk.
            device: Device to use for cache tensor.

        """
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        pos_enc: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        chunk_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Encode input sequences.

        Args:
            x: ConRetformer input sequences. (B, T, D_block)
            mask: Source mask. (B, T)

        Returns:
            mask: Source mask. (B, T)

        """
        residual = x

        x = self.norm_macaron(x)
        x = residual + self.feed_forward_scale * self.dropout(
            self.feed_forward_macaron(x)
        )

        residual = x

        x = self.norm_retention(x)
        x = residual + self.dropout(self.multi_scale_retention(x))

        residual = x

        x = self.norm_conv(x)
        x, _ = self.conv_mod(x, mask=mask)
        x = residual + self.dropout(x)

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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode chunk of input sequence.

        Args:
            x: ConRetformer input sequences. (B, T, D_block)
            mask: Source mask. (B, T_2)

        Returns:
            x: ConRetformer output sequences. (B, T, D_block)

        """
        raise NotImplementedError
