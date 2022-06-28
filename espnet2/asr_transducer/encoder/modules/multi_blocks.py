"""MultiBlocks for encoder architecture."""

from typing import List, Optional, Tuple

import torch


class MultiBlocks(torch.nn.Module):
    """MultiBlocks definition.

    Args:
        block_list: Individual blocks of the encoder architecture.
        output_size: Architecture output size.
        layer_norm: Normalization layer.

    """

    def __init__(
        self,
        block_list: List[torch.nn.Module],
        output_size: int,
        layer_norm: torch.nn.Module = torch.nn.LayerNorm,
    ):
        super().__init__()

        self.blocks = torch.nn.ModuleList(block_list)
        self.norm_blocks = layer_norm(output_size)

        self.num_blocks = len(block_list)

    def init_streaming_cache(self, left_context: int, device: torch.device) -> None:
        """Initialize encoder streaming cache.

        Args:
            left_context:
            device:

        """
        for idx in range(self.num_blocks):
            self.blocks[idx].init_streaming_cache(left_context, device)

    def forward(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor,
        mask: torch.Tensor,
        chunk_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward each block of the encoder architecture.

        Args:
            x:
            pos_enc:
            chunk_mask
            src_mask:

        Returns:
            x:

        """
        for block_index, block in enumerate(self.blocks):
            x = block(x, pos_enc, mask, chunk_mask=chunk_mask)
        x = self.norm_blocks(x)

        return x

    def chunk_forward(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor,
        mask: torch.Tensor,
        left_context: int = 0,
        right_context: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward each block of the encoder architecture.

        Args:
            x:
            pos_enc:
            mask:
            left_context:
            right_context:

        Returns:
            x:
            cache:

        """
        for block_idx, block in enumerate(self.blocks):
            x = block.chunk_forward(
                x,
                pos_enc,
                mask,
                left_context=left_context,
                right_context=right_context,
            )
        x = self.norm_blocks(x)

        return x
