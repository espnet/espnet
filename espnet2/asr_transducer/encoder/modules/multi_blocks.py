"""MultiBlocks for encoder architecture."""

from typing import Dict, List, Optional

import torch


class MultiBlocks(torch.nn.Module):
    """
    MultiBlocks for encoder architecture.

    This class defines a modular approach to building an encoder architecture
    composed of multiple blocks. Each block can be independently designed and
    the architecture supports normalization and block dropout functionality.

    Attributes:
        blocks (torch.nn.ModuleList): A list of blocks constituting the encoder.
        norm_blocks (torch.nn.Module): Normalization module applied to the output.
        blockdrop_rate (float): Probability threshold for dropping each block.
        blockdrop_decay (float): Decay factor for block dropout probabilities.
        keep_probs (torch.Tensor): Tensor holding the probabilities for keeping blocks.

    Args:
        block_list (List[torch.nn.Module]): Individual blocks of the encoder
            architecture.
        output_size (int): Architecture output size.
        norm_class (torch.nn.Module, optional): Normalization module class.
            Defaults to torch.nn.LayerNorm.
        norm_args (Optional[Dict], optional): Normalization module arguments.
            Defaults to None.
        blockdrop_rate (float, optional): Probability threshold for dropping out
            each block. Defaults to 0.0.

    Methods:
        reset_streaming_cache(left_context: int, device: torch.device) -> None:
            Initializes or resets the encoder streaming cache.

        forward(x: torch.Tensor, pos_enc: torch.Tensor, mask: torch.Tensor,
                chunk_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            Forward pass through each block of the encoder architecture.

        chunk_forward(x: torch.Tensor, pos_enc: torch.Tensor, mask: torch.Tensor,
                    left_context: int = 0) -> torch.Tensor:
            Forward pass through each block for chunk processing.

    Examples:
        # Creating a MultiBlocks instance
        blocks = [SomeBlock(), AnotherBlock()]
        multi_blocks = MultiBlocks(block_list=blocks, output_size=256)

        # Forwarding input through the MultiBlocks
        output = multi_blocks(input_tensor, pos_enc_tensor, mask_tensor)

    Note:
        This implementation is designed for use in ASR (Automatic Speech
        Recognition) systems where multiple processing blocks can improve
        model performance.

    Todo:
        - Implement additional normalization strategies.
        - Explore different dropout techniques for blocks.
    """

    def __init__(
        self,
        block_list: List[torch.nn.Module],
        output_size: int,
        norm_class: torch.nn.Module = torch.nn.LayerNorm,
        norm_args: Optional[Dict] = None,
        blockdrop_rate: int = 0.0,
    ) -> None:
        """Construct a MultiBlocks object."""
        super().__init__()

        self.blocks = torch.nn.ModuleList(block_list)
        self.norm_blocks = norm_class(output_size, **norm_args)

        self.blockdrop_rate = blockdrop_rate
        self.blockdrop_decay = 1.0 / len(self.blocks)
        self.keep_probs = torch.ones(len(self.blocks))

    def reset_streaming_cache(self, left_context: int, device: torch.device) -> None:
        """
        Initialize or reset the encoder streaming cache.

        This method is used to prepare the encoder for processing new chunks of
        input by resetting the internal state of each block in the MultiBlocks
        architecture. The `left_context` parameter determines how many previous
        frames the attention module can see in the current chunk, which is
        particularly relevant for architectures like Conformer and Branchformer.

        Args:
            left_context (int): Number of previous frames the attention module
                can see in the current chunk (used by Conformer and Branchformer
                block).
            device (torch.device): Device to use for cache tensor.

        Examples:
            >>> model = MultiBlocks(block_list=[...], output_size=256)
            >>> model.reset_streaming_cache(left_context=5, device=torch.device('cpu'))
        """
        for block in self.blocks:
            block.reset_streaming_cache(left_context, device)

    def forward(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor,
        mask: torch.Tensor,
        chunk_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward each block of the encoder architecture.

        This method processes the input through a series of blocks defined in
        the MultiBlocks architecture. Each block is applied based on a dropout
        probability, allowing for stochastic block dropout during training.
        The output is then normalized using the specified normalization module.

        Args:
            x: MultiBlocks input sequences. Shape: (B, T, D_block_1), where B is
            the batch size, T is the sequence length, and D_block_1 is the
            dimensionality of the first block.
            pos_enc: Positional embedding sequences. Shape: (B, T, D_pos), where
                    D_pos is the dimensionality of the positional encoding.
            mask: Source mask. Shape: (B, T), indicating which elements are
                valid.
            chunk_mask: Optional; Chunk mask. Shape: (T_2, T_2), used for
                        attention masking within chunks.

        Returns:
            x: Output sequences. Shape: (B, T, D_block_N), where D_block_N is
            the dimensionality of the last block after processing.

        Examples:
            >>> multi_blocks = MultiBlocks(block_list=[block1, block2],
            ...                             output_size=128)
            >>> input_tensor = torch.randn(32, 10, 64)  # Batch size 32, T=10, D=64
            >>> pos_tensor = torch.randn(32, 10, 64)
            >>> mask_tensor = torch.ones(32, 10)
            >>> output = multi_blocks.forward(input_tensor, pos_tensor, mask_tensor)

        Note:
            Ensure that the input dimensions match the expected shapes for
            the blocks in the architecture.

        Raises:
            ValueError: If the input tensors do not match the expected
                        dimensions.
        """
        self.keep_probs[:-1].uniform_()

        for idx, block in enumerate(self.blocks):
            if not self.training or (
                self.keep_probs[idx]
                >= (self.blockdrop_rate * (self.blockdrop_decay * idx))
            ):
                x, mask, pos_enc = block(x, pos_enc, mask, chunk_mask=chunk_mask)

        x = self.norm_blocks(x)

        return x

    def chunk_forward(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor,
        mask: torch.Tensor,
        left_context: int = 0,
    ) -> torch.Tensor:
        """
        Forward each block of the encoder architecture.

        This method processes the input tensor `x` through each block in the
        MultiBlocks architecture, applying the `chunk_forward` method of each
        block. It is designed to accommodate the `left_context` parameter, which
        specifies how many previous frames the attention module can see in the
        current chunk. This is particularly useful for models like Conformer and
        Branchformer.

        Args:
            x: MultiBlocks input sequences with shape (B, T, D_block_1).
            pos_enc: Positional embedding sequences with shape (B, 2 * (T - 1), D_att).
            mask: Source mask with shape (B, T_2).
            left_context: Number of previous frames the attention module can see
                        in the current chunk (default is 0).

        Returns:
            x: MultiBlocks output sequences with shape (B, T, D_block_N).

        Examples:
            >>> model = MultiBlocks(block_list=[...], output_size=128)
            >>> x = torch.randn(32, 10, 64)  # Example input
            >>> pos_enc = torch.randn(32, 18, 64)  # Example positional encoding
            >>> mask = torch.ones(32, 10)  # Example mask
            >>> output = model.chunk_forward(x, pos_enc, mask, left_context=2)
            >>> print(output.shape)
            torch.Size([32, 10, 128])

        Note:
            This method is typically used in the context of processing sequences
            where the attention mechanism needs to consider previous frames for
            each chunk.
        """
        for block in self.blocks:
            x, pos_enc = block.chunk_forward(
                x,
                pos_enc,
                mask,
                left_context=left_context,
            )

        x = self.norm_blocks(x)

        return x
