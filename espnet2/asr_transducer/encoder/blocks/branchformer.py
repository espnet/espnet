"""Branchformer block for Transducer encoder."""

from typing import Dict, Optional, Tuple

import torch


class Branchformer(torch.nn.Module):
    """
    Branchformer block for Transducer encoder.

    This class implements the Branchformer module, which is designed to enhance
    the encoding capabilities of a transducer model. It combines self-attention
    and convolutional layers while utilizing normalization and dropout for
    improved performance.

    Reference: https://arxiv.org/pdf/2207.02971.pdf

    Attributes:
        self_att (torch.nn.Module): The self-attention module instance.
        conv_mod (torch.nn.Module): The convolution module instance.
        channel_proj1 (torch.nn.Sequential): A sequential layer for channel
            projection.
        channel_proj2 (torch.nn.Linear): A linear layer for projecting back to
            the original block size.
        merge_proj (torch.nn.Linear): A linear layer for merging outputs from
            attention and convolution.
        norm_self_att (torch.nn.Module): Normalization layer for self-attention.
        norm_mlp (torch.nn.Module): Normalization layer for the MLP.
        norm_final (torch.nn.Module): Final normalization layer.
        dropout (torch.nn.Dropout): Dropout layer for regularization.
        block_size (int): Input/output size.
        linear_size (int): Linear layers' hidden size.
        cache (Optional[List[torch.Tensor]]): Cache for storing intermediate
            results during streaming.

    Args:
        block_size (int): Input/output size.
        linear_size (int): Linear layers' hidden size.
        self_att (torch.nn.Module): Self-attention module instance.
        conv_mod (torch.nn.Module): Convolution module instance.
        norm_class (torch.nn.Module, optional): Normalization class. Defaults to
            torch.nn.LayerNorm.
        norm_args (Dict, optional): Normalization module arguments. Defaults to {}.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.0.

    Examples:
        # Initialize a Branchformer module
        branchformer = Branchformer(
            block_size=256,
            linear_size=512,
            self_att=my_self_attention_module,
            conv_mod=my_convolution_module,
            norm_class=torch.nn.LayerNorm,
            norm_args={'eps': 1e-6},
            dropout_rate=0.1
        )

        # Forward pass
        output, mask, pos_enc = branchformer(x, pos_enc, mask)

        # Reset cache for streaming
        branchformer.reset_streaming_cache(left_context=10, device=torch.device('cuda'))

    Raises:
        ValueError: If the input tensor dimensions do not match the expected sizes.
    """

    def __init__(
        self,
        block_size: int,
        linear_size: int,
        self_att: torch.nn.Module,
        conv_mod: torch.nn.Module,
        norm_class: torch.nn.Module = torch.nn.LayerNorm,
        norm_args: Dict = {},
        dropout_rate: float = 0.0,
    ) -> None:
        """Construct a Branchformer object."""
        super().__init__()

        self.self_att = self_att
        self.conv_mod = conv_mod

        self.channel_proj1 = torch.nn.Sequential(
            torch.nn.Linear(block_size, linear_size), torch.nn.GELU()
        )
        self.channel_proj2 = torch.nn.Linear(linear_size // 2, block_size)

        self.merge_proj = torch.nn.Linear(block_size + block_size, block_size)

        self.norm_self_att = norm_class(block_size, **norm_args)
        self.norm_mlp = norm_class(block_size, **norm_args)
        self.norm_final = norm_class(block_size, **norm_args)

        self.dropout = torch.nn.Dropout(dropout_rate)

        self.block_size = block_size
        self.linear_size = linear_size
        self.cache = None

    def reset_streaming_cache(self, left_context: int, device: torch.device) -> None:
        """
        Initialize/Reset self-attention and convolution modules cache for streaming.

        This method resets the internal cache used by the self-attention and
        convolution modules. It creates new tensors to hold the cached values
        for both attention and convolution, which are essential for processing
        streaming data.

        Args:
            left_context: Number of previous frames the attention module can see
                          in the current chunk. This defines how much context
                          is available for attention calculations.
            device: Device to use for cache tensor. This specifies where the
                    cache tensors should be allocated (e.g., CPU or GPU).

        Examples:
            >>> model = Branchformer(...)
            >>> model.reset_streaming_cache(left_context=10, device=torch.device('cuda'))

        Note:
            This method should be called whenever the input context changes or
            when starting a new streaming session to ensure the cache is
            appropriately initialized.
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
        ]

    def forward(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor,
        mask: torch.Tensor,
        chunk_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Branchformer block for Transducer encoder.

        This module implements the Branchformer architecture, which is designed for
        efficient sequence encoding using self-attention and convolutional mechanisms.

        Reference:
            https://arxiv.org/pdf/2207.02971.pdf

        Args:
            block_size (int): Input/output size.
            linear_size (int): Linear layers' hidden size.
            self_att (torch.nn.Module): Self-attention module instance.
            conv_mod (torch.nn.Module): Convolution module instance.
            norm_class (torch.nn.Module, optional): Normalization class, defaults to
                torch.nn.LayerNorm.
            norm_args (Dict, optional): Normalization module arguments, defaults to {}.
            dropout_rate (float, optional): Dropout rate, defaults to 0.0.

        Methods:
            reset_streaming_cache(left_context: int, device: torch.device) -> None:
                Initializes or resets the self-attention and convolution modules' cache
                for streaming.

            forward(
                x: torch.Tensor,
                pos_enc: torch.Tensor,
                mask: torch.Tensor,
                chunk_mask: Optional[torch.Tensor] = None
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                Encodes input sequences.

            chunk_forward(
                x: torch.Tensor,
                pos_enc: torch.Tensor,
                mask: torch.Tensor,
                left_context: int = 0
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                Encodes a chunk of the input sequence.

        Examples:
            # Example usage of the Branchformer module
            branchformer = Branchformer(
                block_size=256,
                linear_size=512,
                self_att=torch.nn.MultiheadAttention(embed_dim=256, num_heads=8),
                conv_mod=torch.nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3)
            )

            x = torch.randn(32, 10, 256)  # (B, T, D_block)
            pos_enc = torch.randn(32, 18, 256)  # (B, 2 * (T - 1), D_block)
            mask = torch.ones(32, 10)  # (B, T)

            output, mask_out, pos_enc_out = branchformer.forward(x, pos_enc, mask)
        """
        x1 = x
        x2 = x

        x1 = self.norm_self_att(x1)

        x1 = self.dropout(
            self.self_att(x1, x1, x1, pos_enc, mask=mask, chunk_mask=chunk_mask)
        )

        x2 = self.norm_mlp(x2)

        x2 = self.channel_proj1(x2)
        x2, _ = self.conv_mod(x2, mask)
        x2 = self.channel_proj2(x2)

        x2 = self.dropout(x2)

        x = x + self.dropout(self.merge_proj(torch.cat([x1, x2], dim=-1)))

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
            x: Branchformer input sequences. (B, T, D_block)
            pos_enc: Positional embedding sequences. (B, 2 * (T - 1), D_block)
            mask: Source mask. (B, T_2)
            left_context: Number of previous frames the attention module can see
                          in current chunk.

        Returns:
            x: Branchformer output sequences. (B, T, D_block)
            pos_enc: Positional embedding sequences. (B, 2 * (T - 1), D_block)

        """
        x1 = x
        x2 = x

        x1 = self.norm_self_att(x1)

        if left_context > 0:
            key = torch.cat([self.cache[0], x1], dim=1)
        else:
            key = x1
        val = key

        att_cache = key[:, -left_context:, :]
        x1 = self.self_att(x1, key, val, pos_enc, mask=mask, left_context=left_context)

        x2 = self.norm_mlp(x2)

        x2 = self.channel_proj1(x2)
        x2, conv_cache = self.conv_mod(x2, cache=self.cache[1])
        x2 = self.channel_proj2(x2)

        x = x + self.merge_proj(torch.cat([x1, x2], dim=-1))

        x = self.norm_final(x)

        self.cache = [att_cache, conv_cache]

        return x, pos_enc
