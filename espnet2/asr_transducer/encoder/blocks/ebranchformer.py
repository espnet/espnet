"""E-Branchformer block for Transducer encoder."""

from typing import Dict, Optional, Tuple

import torch


class EBranchformer(torch.nn.Module):
    """
    E-Branchformer block for Transducer encoder.

    This class implements the E-Branchformer module, which is a crucial component
    of the Transducer encoder architecture. It incorporates self-attention,
    feed-forward networks, and convolutional layers to process input sequences
    effectively.

    Reference: https://arxiv.org/pdf/2210.00077.pdf

    Attributes:
        self_att (torch.nn.Module): Instance of the self-attention module.
        feed_forward (torch.nn.Module): Instance of the feed-forward module.
        feed_forward_macaron (torch.nn.Module): Instance of the macaron feed-forward module.
        conv_mod (torch.nn.Module): Instance of the ConvolutionalSpatialGatingUnit module.
        depthwise_conv_mod (torch.nn.Module): Instance of the DepthwiseConvolution module.
        norm_self_att (torch.nn.Module): Normalization layer for self-attention.
        norm_feed_forward (torch.nn.Module): Normalization layer for feed-forward network.
        norm_feed_forward_macaron (torch.nn.Module): Normalization layer for macaron feed-forward.
        norm_mlp (torch.nn.Module): Normalization layer for MLP.
        norm_final (torch.nn.Module): Final normalization layer.
        dropout (torch.nn.Dropout): Dropout layer for regularization.
        block_size (int): Input/output size of the block.
        linear_size (int): Hidden size of the linear layers.
        cache (Optional[List[torch.Tensor]]): Cache for self-attention and convolution.

    Args:
        block_size (int): Input/output size.
        linear_size (int): Linear layers' hidden size.
        self_att (torch.nn.Module): Self-attention module instance.
        feed_forward (torch.nn.Module): Feed-forward module instance.
        feed_forward_macaron (torch.nn.Module): Feed-forward module instance for macaron network.
        conv_mod (torch.nn.Module): ConvolutionalSpatialGatingUnit module instance.
        depthwise_conv_mod (torch.nn.Module): DepthwiseConvolution module instance.
        norm_class (torch.nn.Module, optional): Normalization class (default: LayerNorm).
        norm_args (Dict, optional): Normalization module arguments (default: {}).
        dropout_rate (float, optional): Dropout rate (default: 0.0).

    Examples:
        # Create an E-Branchformer instance
        e_branchformer = EBranchformer(
            block_size=128,
            linear_size=256,
            self_att=my_self_att_module,
            feed_forward=my_feed_forward_module,
            feed_forward_macaron=my_macaron_feed_forward_module,
            conv_mod=my_conv_module,
            depthwise_conv_mod=my_depthwise_conv_module,
            norm_class=torch.nn.LayerNorm,
            norm_args={'eps': 1e-5},
            dropout_rate=0.1
        )

        # Forward pass through the E-Branchformer
        output, mask, pos_enc = e_branchformer(input_tensor, pos_enc_tensor, mask_tensor)

        # Resetting the streaming cache
        e_branchformer.reset_streaming_cache(left_context=10, device=torch.device('cuda'))

    Note:
        The module requires specific instances of self-attention, feed-forward,
        convolution, and normalization modules. These modules should be defined
        prior to instantiation of the EBranchformer class.

    Todo:
        - Implement additional features for enhanced performance and efficiency.
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
        """
        Initialize/Reset self-attention and convolution modules cache for streaming.

        This method initializes or resets the cache for the self-attention and
        convolution modules to facilitate streaming processing. The cache is used
        to store intermediate results from previous frames, allowing the model to
        maintain context over longer sequences without having to recompute past
        information.

        Args:
            left_context: Number of previous frames the attention module can see
                          in current chunk. This parameter controls how much of
                          the past context is considered during attention
                          calculations.
            device: Device to use for cache tensor. This allows the cache to be
                    created on the appropriate hardware (CPU or GPU) for
                    efficient processing.

        Examples:
            >>> model = EBranchformer(...)
            >>> model.reset_streaming_cache(left_context=5, device=torch.device('cuda'))

        Note:
            This method should be called before processing a new input chunk to
            ensure that the cache is properly initialized.
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
        """
        Encode input sequences using the E-Branchformer module.

        The forward method processes input sequences through various layers,
        including self-attention and feed-forward layers, to produce the
        output sequences. The method also supports masking for attention
        mechanisms and incorporates residual connections for better
        gradient flow.

        Args:
            x (torch.Tensor): E-Branchformer input sequences of shape
                            (B, T, D_block), where B is the batch size,
                            T is the sequence length, and D_block is
                            the dimensionality of the block.
            pos_enc (torch.Tensor): Positional embedding sequences of shape
                                    (B, 2 * (T - 1), D_block), representing
                                    the position of each token in the sequence.
            mask (torch.Tensor): Source mask of shape (B, T) to specify which
                                tokens should be attended to.
            chunk_mask (Optional[torch.Tensor]): Optional chunk mask of shape
                                                (T_2, T_2) to control
                                                attention within chunks.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - x (torch.Tensor): E-Branchformer output sequences of shape
                                    (B, T, D_block).
                - mask (torch.Tensor): Source mask of shape (B, T).
                - pos_enc (torch.Tensor): Positional embedding sequences of shape
                                        (B, 2 * (T - 1), D_block).

        Examples:
            >>> model = EBranchformer(block_size=256, linear_size=128,
            ...                        self_att=self_attention_module,
            ...                        feed_forward=feed_forward_module,
            ...                        feed_forward_macaron=feed_forward_macaron_module,
            ...                        conv_mod=conv_module,
            ...                        depthwise_conv_mod=depthwise_conv_module)
            >>> input_tensor = torch.randn(32, 10, 256)  # Batch of 32, 10 time steps
            >>> pos_enc = torch.randn(32, 18, 256)  # Positional encodings
            >>> mask = torch.ones(32, 10)  # Full attention
            >>> output, mask_out, pos_enc_out = model.forward(input_tensor, pos_enc, mask)
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
        """
        Encode chunk of input sequence.

        This method processes a chunk of the input sequence through the E-Branchformer
        architecture, incorporating self-attention and feed-forward mechanisms while
        considering the specified left context for attention.

        Args:
            x: E-Branchformer input sequences. Shape: (B, T, D_block), where
               B is the batch size, T is the sequence length, and D_block is the
               dimensionality of the input features.
            pos_enc: Positional embedding sequences. Shape: (B, 2 * (T - 1), D_block).
            mask: Source mask. Shape: (B, T_2), used to prevent attention to
                  certain positions in the input.
            left_context: Number of previous frames the attention module can see
                          in the current chunk. Defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - x: E-Branchformer output sequences. Shape: (B, T, D_block).
                - pos_enc: Positional embedding sequences. Shape: (B, 2 * (T - 1), D_block).

        Examples:
            >>> e_branchformer = EBranchformer(block_size=128, linear_size=256,
            ...                                 self_att=my_self_att,
            ...                                 feed_forward=my_feed_forward,
            ...                                 feed_forward_macaron=my_feed_forward_macaron,
            ...                                 conv_mod=my_conv_mod,
            ...                                 depthwise_conv_mod=my_depthwise_conv_mod)
            >>> x = torch.randn(10, 20, 128)  # Batch of 10, sequence length 20
            >>> pos_enc = torch.randn(10, 38, 128)  # Positional encoding for T-1
            >>> mask = torch.ones(10, 20)  # Full attention mask
            >>> output, pos_enc_out = e_branchformer.chunk_forward(x, pos_enc, mask)

        Note:
            This method maintains a cache for self-attention and convolutional
            layers to optimize processing of sequential data.
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
