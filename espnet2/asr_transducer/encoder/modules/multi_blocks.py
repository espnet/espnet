"""MultiBlocks for encoder architecture."""

from typing import Dict, List, Optional

import torch


class MultiBlocks(torch.nn.Module):
    """MultiBlocks definition.

    Args:
        block_list: Individual blocks of the encoder architecture.
        output_size: Architecture output size.
        norm_class: Normalization module class.
        norm_args: Normalization module arguments.

    """

    def __init__(
        self,
        block_list: List[torch.nn.Module],
        output_size: int,
        norm_class: torch.nn.Module = torch.nn.LayerNorm,
        norm_args: Optional[Dict] = None,
    ) -> None:
        """Construct a MultiBlocks object."""
        super().__init__()

        self.blocks = torch.nn.ModuleList(block_list)
        self.norm_blocks = norm_class(output_size, **norm_args)

        self.num_blocks = len(block_list)

    def reset_streaming_cache(self, left_context: int, device: torch.device) -> None:
        """Initialize/Reset encoder streaming cache.

        Args:
            left_context: Number of left frames during chunk-by-chunk inference.
            device: Device to use for cache tensor.

        """
        for idx in range(self.num_blocks):
            self.blocks[idx].reset_streaming_cache(left_context, device)

    def forward(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor,
        mask: torch.Tensor,
        chunk_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward each block of the encoder architecture.

        Args:
            x: MultiBlocks input sequences. (B, T, D_block_1)
            pos_enc: Positional embedding sequences.
            mask: Source mask. (B, T)
            chunk_mask: Chunk mask. (T_2, T_2)

        Returns:
            x: Output sequences. (B, T, D_block_N)

        """
        for block_index, block in enumerate(self.blocks):
            x, mask, pos_enc = block(x, pos_enc, mask, chunk_mask=chunk_mask)

        x = self.norm_blocks(x)

        return x

    def chunk_forward(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor,
        mask: torch.Tensor,
        left_context: int = 0,
        right_context: int = 0,
    ) -> torch.Tensor:
        """Forward each block of the encoder architecture.

        Args:
            x: MultiBlocks input sequences. (B, T, D_block_1)
            pos_enc: Positional embedding sequences. (B, 2 * (T - 1), D_att)
            mask: Source mask. (B, T_2)
            left_context: Number of frames in left context.
            right_context: Number of frames in right context.

        Returns:
            x: MultiBlocks output sequences. (B, T, D_block_N)

        """
        for block_idx, block in enumerate(self.blocks):
            x, pos_enc = block.chunk_forward(
                x,
                pos_enc,
                mask,
                left_context=left_context,
                right_context=right_context,
            )

        x = self.norm_blocks(x)

        return x
