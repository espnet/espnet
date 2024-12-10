# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""
from typing import Optional, Tuple

import torch
from typeguard import typechecked

from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import (
    Conv1dLinear,
    MultiLayeredConv1d,
)
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat


class TransformerPostEncoder(AbsPostEncoder):
    """
    Transformer encoder module for sequence-to-sequence tasks.

    This module implements a transformer encoder as part of the post-encoder
    architecture for various sequence-to-sequence tasks. It consists of a
    stack of encoder layers, each employing multi-head attention and
    position-wise feed-forward networks.

    Attributes:
        output_size (int): The dimension of the output features.
        embed (torch.nn.Sequential): The embedding layer that includes
            linear transformation, normalization, dropout, and positional
            encoding.
        encoders (torch.nn.ModuleList): A list of encoder layers that
            process the input embeddings.
        after_norm (LayerNorm, optional): Layer normalization applied after
            the encoder layers if `normalize_before` is True.

    Args:
        input_size (int): The dimensionality of the input features.
        output_size (int): The dimensionality of the output features
            (default: 256).
        attention_heads (int): The number of heads in the multi-head
            attention (default: 4).
        linear_units (int): The number of units in the position-wise feed
            forward layer (default: 2048).
        num_blocks (int): The number of encoder blocks (default: 6).
        dropout_rate (float): Dropout rate applied to layers (default: 0.1).
        positional_dropout_rate (float): Dropout rate after adding
            positional encoding (default: 0.1).
        attention_dropout_rate (float): Dropout rate applied within the
            attention layers (default: 0.0).
        input_layer (Optional[str]): Type of input layer; either "linear"
            or "None" (default: "linear").
        pos_enc_class: Class for positional encoding; typically
            PositionalEncoding or ScaledPositionalEncoding.
        normalize_before (bool): Whether to apply layer normalization before
            the first encoder block (default: True).
        concat_after (bool): If True, concatenates input and output of the
            attention layer, followed by a linear transformation (default: False).
        positionwise_layer_type (str): Type of position-wise layer;
            "linear", "conv1d", or "conv1d-linear" (default: "linear").
        positionwise_conv_kernel_size (int): Kernel size for position-wise
            convolutional layer (default: 1).
        padding_idx (int): Padding index for the input layer when
            `input_layer` is "embed" (default: -1).

    Examples:
        >>> encoder = TransformerPostEncoder(
        ...     input_size=128,
        ...     output_size=256,
        ...     attention_heads=4,
        ...     linear_units=2048,
        ...     num_blocks=6
        ... )
        >>> xs_pad = torch.rand(10, 20, 128)  # (B, L, D)
        >>> ilens = torch.tensor([20] * 10)    # (B)
        >>> output, olens = encoder(xs_pad, ilens)
        >>> print(output.shape)  # (B, L, output_size)
        >>> print(olens.shape)   # (B,)

    Note:
        The input tensor `xs_pad` should be padded appropriately, and the
        lengths of the sequences should be provided in `ilens`.

    Raises:
        ValueError: If an unknown `input_layer` type is specified.
        NotImplementedError: If an unsupported `positionwise_layer_type`
            is specified.
    """

    @typechecked
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "linear",
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        padding_idx: int = -1,
    ):
        super().__init__()
        self._output_size = output_size

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "None":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")
        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                output_size,
                MultiHeadedAttention(
                    attention_heads, output_size, attention_dropout_rate
                ),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

    def output_size(self) -> int:
        """
        Return the output size of the TransformerPostEncoder.

        This method provides the dimension of the output from the encoder
        layers. The output size is defined during the initialization of the
        TransformerPostEncoder class and remains constant throughout its
        lifetime.

        Returns:
            int: The output size, which corresponds to the dimension of the
            attention layer.

        Examples:
            >>> transformer = TransformerPostEncoder(input_size=128, output_size=256)
            >>> transformer.output_size()
            256
        """
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Process the input tensor through the Transformer encoder.

        This method applies position embedding, processes the input through
        multiple encoder layers, and optionally normalizes the output.

        Args:
            xs_pad (torch.Tensor): Input tensor of shape (B, L, D), where B
                is the batch size, L is the sequence length, and D is the
                dimension of the input features.
            ilens (torch.Tensor): Tensor containing the lengths of each input
                sequence in the batch of shape (B).
            prev_states (torch.Tensor, optional): Previous states (not used
                currently). Default is None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                - Processed tensor of shape (B, L, D) after embedding and
                  encoding.
                - Tensor containing the output lengths of each sequence in the
                  batch of shape (B).
                - Optional tensor (not used currently) for any additional
                  states.

        Examples:
            >>> model = TransformerPostEncoder(input_size=128)
            >>> xs_pad = torch.randn(32, 50, 128)  # Batch of 32 sequences
            >>> ilens = torch.tensor([50] * 32)  # All sequences of length 50
            >>> output, olens = model.forward(xs_pad, ilens)
            >>> print(output.shape)  # Output shape should be (32, 50, 256)
            >>> print(olens.shape)  # Output lengths shape should be (32,)

        Note:
            This method expects the input tensor to be padded. Ensure that
            the `ilens` tensor accurately reflects the lengths of the sequences
            in `xs_pad`.

        Raises:
            ValueError: If `ilens` is not a tensor or has an unexpected shape.
        """
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

        xs_pad = self.embed(xs_pad)
        xs_pad, masks = self.encoders(xs_pad, masks)
        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        olens = masks.squeeze(1).sum(1)
        return xs_pad, olens
