"""Conformer block for Transducer encoder."""

from typing import Dict, Optional, Tuple

import torch


class Conformer(torch.nn.Module):
    """
    Conformer block for Transducer encoder.

    This module implements a Conformer block, which is a type of neural network
    architecture that combines convolutional layers and self-attention mechanisms.
    It is designed to be used within an automatic speech recognition (ASR)
    transducer encoder.

    Attributes:
        self_att (torch.nn.Module): Self-attention module instance.
        feed_forward (torch.nn.Module): Feed-forward module instance.
        feed_forward_macaron (torch.nn.Module): Feed-forward module instance for
            macaron network.
        conv_mod (torch.nn.Module): Convolution module instance.
        norm_feed_forward (torch.nn.Module): Normalization module for the feed-forward
            component.
        norm_self_att (torch.nn.Module): Normalization module for the self-attention
            component.
        norm_macaron (torch.nn.Module): Normalization module for the macaron feed-forward
            component.
        norm_conv (torch.nn.Module): Normalization module for the convolution component.
        norm_final (torch.nn.Module): Final normalization module.
        dropout (torch.nn.Dropout): Dropout layer for regularization.
        block_size (int): Input/output size of the block.
        cache (Optional[List[torch.Tensor]]): Caches for self-attention and
            convolution modules during streaming.

    Args:
        block_size (int): Input/output size.
        self_att (torch.nn.Module): Self-attention module instance.
        feed_forward (torch.nn.Module): Feed-forward module instance.
        feed_forward_macaron (torch.nn.Module): Feed-forward module instance for
            macaron network.
        conv_mod (torch.nn.Module): Convolution module instance.
        norm_class (torch.nn.Module, optional): Normalization module class.
            Defaults to torch.nn.LayerNorm.
        norm_args (Dict, optional): Normalization module arguments. Defaults to {}.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.0.

    Methods:
        reset_streaming_cache(left_context: int, device: torch.device) -> None:
            Initialize/Reset self-attention and convolution modules cache for streaming.
        forward(x: torch.Tensor, pos_enc: torch.Tensor, mask: torch.Tensor,
            chunk_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor,
            torch.Tensor, torch.Tensor]:
            Encode input sequences.
        chunk_forward(x: torch.Tensor, pos_enc: torch.Tensor, mask: torch.Tensor,
            left_context: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
            Encode chunk of input sequence.

    Examples:
        # Example of creating a Conformer block
        conformer_block = Conformer(
            block_size=256,
            self_att=SelfAttentionModule(),
            feed_forward=FeedForwardModule(),
            feed_forward_macaron=FeedForwardModule(),
            conv_mod=ConvolutionModule(),
            norm_class=torch.nn.LayerNorm,
            norm_args={'eps': 1e-5},
            dropout_rate=0.1
        )

        # Example of using the forward method
        output, mask, pos_enc = conformer_block.forward(
            x=torch.randn(10, 20, 256),  # Batch size 10, sequence length 20, feature size 256
            pos_enc=torch.randn(10, 38, 256),  # Positional encoding
            mask=torch.ones(10, 20)  # Source mask
        )

        # Example of using the reset_streaming_cache method
        conformer_block.reset_streaming_cache(left_context=5, device='cuda')

        # Example of using the chunk_forward method
        chunk_output, updated_pos_enc = conformer_block.chunk_forward(
            x=torch.randn(10, 15, 256),  # Batch size 10, sequence length 15, feature size 256
            pos_enc=torch.randn(10, 30, 256),  # Positional encoding
            mask=torch.ones(10, 15),  # Source mask
            left_context=5
        )
    """

    def __init__(
        self,
        block_size: int,
        self_att: torch.nn.Module,
        feed_forward: torch.nn.Module,
        feed_forward_macaron: torch.nn.Module,
        conv_mod: torch.nn.Module,
        norm_class: torch.nn.Module = torch.nn.LayerNorm,
        norm_args: Dict = {},
        dropout_rate: float = 0.0,
    ) -> None:
        """Construct a Conformer object."""
        super().__init__()

        self.self_att = self_att

        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.feed_forward_scale = 0.5

        self.conv_mod = conv_mod

        self.norm_feed_forward = norm_class(block_size, **norm_args)
        self.norm_self_att = norm_class(block_size, **norm_args)

        self.norm_macaron = norm_class(block_size, **norm_args)
        self.norm_conv = norm_class(block_size, **norm_args)
        self.norm_final = norm_class(block_size, **norm_args)

        self.dropout = torch.nn.Dropout(dropout_rate)

        self.block_size = block_size
        self.cache = None

    def reset_streaming_cache(self, left_context: int, device: torch.device) -> None:
        """
        Initialize or reset the streaming cache for self-attention and
        convolution modules.

        This method sets up the cache used by the self-attention and
        convolution layers to facilitate streaming processing of input
        sequences. The cache is initialized based on the specified
        `left_context`, which determines how many previous frames the
        attention module can access for the current chunk of input data.

        Args:
            left_context: An integer representing the number of previous
                          frames that the attention module can see in the
                          current chunk.
            device: The device (CPU or GPU) to use for creating the cache
                    tensors.

        Examples:
            >>> conformer = Conformer(...)
            >>> conformer.reset_streaming_cache(left_context=5, device=torch.device('cuda'))

        Note:
            The cache is a list containing two tensors: the first tensor
            holds the self-attention cache, and the second tensor holds
            the convolution cache. The shape of the tensors is determined
            by the `block_size` and the convolution module's kernel size.
        """
        self.cache = [
            torch.zeros(
                (1, left_context, self.block_size),
                device=device,
            ),
            torch.zeros(
                (
                    1,
                    self.block_size,
                    self.conv_mod.kernel_size - 1,
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
        """
        Encode input sequences through the Conformer module.

        This method processes the input sequences using self-attention, convolution,
        and feed-forward layers, applying normalization and dropout as specified.

        Args:
            x (torch.Tensor): Conformer input sequences of shape (B, T, D_block).
            pos_enc (torch.Tensor): Positional embedding sequences of shape
                                    (B, 2 * (T - 1), D_block).
            mask (torch.Tensor): Source mask of shape (B, T).
            chunk_mask (Optional[torch.Tensor]): Optional chunk mask of shape
                                                (T_2, T_2).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - x (torch.Tensor): Conformer output sequences of shape (B, T, D_block).
                - mask (torch.Tensor): Source mask of shape (B, T).
                - pos_enc (torch.Tensor): Positional embedding sequences of shape
                                        (B, 2 * (T - 1), D_block).

        Examples:
            >>> model = Conformer(block_size=128,
            ...                   self_att=self_attention_module,
            ...                   feed_forward=feed_forward_module,
            ...                   feed_forward_macaron=feed_forward_macaron_module,
            ...                   conv_mod=conv_module)
            >>> input_sequences = torch.randn(32, 10, 128)
            >>> positional_encodings = torch.randn(32, 18, 128)
            >>> source_mask = torch.ones(32, 10)
            >>> output, mask, pos_enc = model(input_sequences, positional_encodings,
            ...                                source_mask)

        Note:
            This method is designed to be used within the context of the Conformer
            architecture and requires the input tensors to be appropriately shaped
            and normalized before being passed in.
        """
        residual = x

        x = self.norm_macaron(x)
        x = residual + self.feed_forward_scale * self.dropout(
            self.feed_forward_macaron(x)
        )

        residual = x
        x = self.norm_self_att(x)

        x = residual + self.dropout(
            self.self_att(
                x,
                x,
                x,
                pos_enc,
                mask,
                chunk_mask=chunk_mask,
            )
        )

        residual = x

        x = self.norm_conv(x)
        x, _ = self.conv_mod(x, mask=mask)
        x = residual + self.dropout(x)

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
        """
        Encode chunk of input sequence.

        This method processes a chunk of the input sequence, utilizing the
        self-attention mechanism while considering a specified number of
        previous frames as context. It updates the internal cache for
        streaming purposes, allowing the model to maintain state across
        chunks.

        Args:
            x: Conformer input sequences. Shape: (B, T, D_block)
            pos_enc: Positional embedding sequences. Shape: (B, 2 * (T - 1), D_block)
            mask: Source mask. Shape: (B, T_2)
            left_context: Number of previous frames the attention module can
                        see in the current chunk. Default is 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                x: Conformer output sequences. Shape: (B, T, D_block)
                pos_enc: Positional embedding sequences. Shape: (B, 2 * (T - 1), D_block)

        Examples:
            >>> conformer = Conformer(block_size=128, self_att=..., feed_forward=...,
            ... feed_forward_macaron=..., conv_mod=..., norm_class=torch.nn.LayerNorm)
            >>> x = torch.randn(10, 20, 128)  # Batch size 10, sequence length 20
            >>> pos_enc = torch.randn(10, 38, 128)  # Positional encodings
            >>> mask = torch.ones(10, 20)  # Source mask
            >>> output, updated_pos_enc = conformer.chunk_forward(x, pos_enc, mask,
            ... left_context=5)

        Note:
            The method modifies the internal cache, which is used to retain
            information from previous chunks, enhancing the performance in
            streaming scenarios.
        """
        residual = x

        x = self.norm_macaron(x)
        x = residual + self.feed_forward_scale * self.feed_forward_macaron(x)

        residual = x
        x = self.norm_self_att(x)

        if left_context > 0:
            key = torch.cat([self.cache[0], x], dim=1)
        else:
            key = x
        att_cache = key[:, -left_context:, :]

        x = residual + self.self_att(
            x,
            key,
            key,
            pos_enc,
            mask,
            left_context=left_context,
        )

        residual = x

        x = self.norm_conv(x)
        x, conv_cache = self.conv_mod(x, cache=self.cache[1])
        x = residual + x

        residual = x

        x = self.norm_feed_forward(x)
        x = residual + self.feed_forward_scale * self.feed_forward(x)

        x = self.norm_final(x)

        self.cache = [att_cache, conv_cache]

        return x, pos_enc
