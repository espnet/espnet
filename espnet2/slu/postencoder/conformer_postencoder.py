#!/usr/bin/env python3
#  2021, Carnegie Mellon University;  Siddhant Arora
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Conformers PostEncoder."""
import logging
from typing import Tuple

import torch
from typeguard import typechecked

from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet.nets.pytorch_backend.conformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.conformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.nets_utils import get_activation, make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import (  # noqa: H301
    LegacyRelPositionMultiHeadedAttention,
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from espnet.nets.pytorch_backend.transformer.embedding import (  # noqa: H301
    LegacyRelPositionalEncoding,
    PositionalEncoding,
    RelPositionalEncoding,
    ScaledPositionalEncoding,
)
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import (
    Conv1dLinear,
    MultiLayeredConv1d,
)
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat


class ConformerPostEncoder(AbsPostEncoder):
    """
    Conformer PostEncoder for sequence-to-sequence models.

    This class implements a Conformer encoder module that processes input
    sequences using multi-head attention, convolutional layers, and
    position-wise feed-forward networks. It is designed to enhance the
    performance of speech and language processing tasks by capturing
    contextual information effectively.

    Attributes:
        output_size (int): The output dimension of the encoder.
        embed (torch.nn.Sequential): The embedding layer that includes
            positional encoding.
        normalize_before (bool): Flag to indicate if layer normalization
            is applied before the first block.
        encoders (torch.nn.ModuleList): A list of encoder layers.
        after_norm (torch.nn.LayerNorm): Layer normalization applied after
            the encoder if normalize_before is True.

    Args:
        input_size (int): Input dimension.
        output_size (int): Dimension of attention (default: 256).
        attention_heads (int): Number of heads in multi-head attention
            (default: 4).
        linear_units (int): Number of units in position-wise feed forward
            (default: 2048).
        num_blocks (int): Number of decoder blocks (default: 6).
        dropout_rate (float): Dropout rate (default: 0.1).
        attention_dropout_rate (float): Dropout rate in attention
            (default: 0.0).
        positional_dropout_rate (float): Dropout rate after adding
            positional encoding (default: 0.1).
        input_layer (Union[str, torch.nn.Module]): Input layer type
            (default: "linear").
        normalize_before (bool): Whether to use layer normalization before
            the first block (default: True).
        concat_after (bool): Whether to concatenate input and output of
            attention layer (default: False).
        positionwise_layer_type (str): Type of position-wise layer
            ("linear", "conv1d", or "conv1d-linear", default: "linear").
        positionwise_conv_kernel_size (int): Kernel size for position-wise
            convolution (default: 3).
        rel_pos_type (str): Type of relative positional encoding
            ("legacy" or "latest", default: "legacy").
        encoder_pos_enc_layer_type (str): Type of encoder positional
            encoding layer (default: "rel_pos").
        encoder_attn_layer_type (str): Type of encoder attention layer
            (default: "selfattn").
        activation_type (str): Activation function type (default: "swish").
        macaron_style (bool): Whether to use Macaron style for position-wise
            layer (default: False).
        use_cnn_module (bool): Whether to use convolution module
            (default: True).
        zero_triu (bool): Whether to zero the upper triangular part of the
            attention matrix (default: False).
        cnn_module_kernel (int): Kernel size of convolution module
            (default: 31).
        padding_idx (int): Padding index for input_layer=embed (default: -1).

    Raises:
        ValueError: If unknown values are provided for `rel_pos_type`,
            `pos_enc_layer_type`, or `input_layer`.

    Examples:
        >>> encoder = ConformerPostEncoder(input_size=128)
        >>> input_tensor = torch.randn(10, 20, 128)  # (batch_size, seq_len, feature_dim)
        >>> input_lengths = torch.tensor([20] * 10)  # All sequences are of length 20
        >>> output, output_lengths = encoder(input_tensor, input_lengths)
        >>> print(output.shape)  # Should match (batch_size, seq_len, output_size)
        >>> print(output_lengths.shape)  # Should match (batch_size,)
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
        input_layer: str = "linear",
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 3,
        macaron_style: bool = False,
        rel_pos_type: str = "legacy",
        pos_enc_layer_type: str = "rel_pos",
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        zero_triu: bool = False,
        cnn_module_kernel: int = 31,
        padding_idx: int = -1,
    ):
        super().__init__()
        self._output_size = output_size

        if rel_pos_type == "legacy":
            if pos_enc_layer_type == "rel_pos":
                pos_enc_layer_type = "legacy_rel_pos"
            if selfattention_layer_type == "rel_selfattn":
                selfattention_layer_type = "legacy_rel_selfattn"
        elif rel_pos_type == "latest":
            assert selfattention_layer_type != "legacy_rel_selfattn"
            assert pos_enc_layer_type != "legacy_rel_pos"
        else:
            raise ValueError("unknown rel_pos_type: " + rel_pos_type)

        activation = get_activation(activation_type)
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert selfattention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "legacy_rel_pos":
            assert selfattention_layer_type == "legacy_rel_selfattn"
            pos_enc_class = LegacyRelPositionalEncoding
            logging.warning(
                "Using legacy_rel_pos and it will be deprecated in the future."
            )
        elif pos_enc_layer_type == "None":
            pos_enc_class = None
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "None":
            self.embed = None
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
                activation,
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

        if selfattention_layer_type == "selfattn":
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
        elif selfattention_layer_type == "legacy_rel_selfattn":
            assert pos_enc_layer_type == "legacy_rel_pos"
            encoder_selfattn_layer = LegacyRelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
            logging.warning(
                "Using legacy_rel_selfattn and it will be deprecated in the future."
            )
        elif selfattention_layer_type == "rel_selfattn":
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
                zero_triu,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + selfattention_layer_type)

        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation)

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the ConformerPostEncoder.

        This method processes the input tensor through the embedding layer,
        followed by multiple encoder layers. It applies masking to handle
        padded sequences and normalizes the output if specified.

        Args:
            input (torch.Tensor): The input tensor of shape (batch_size,
                sequence_length, feature_dim).
            input_lengths (torch.Tensor): A tensor of shape (batch_size,)
                containing the lengths of each input sequence before padding.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - output (torch.Tensor): The processed output tensor of shape
                  (batch_size, sequence_length, output_dim).
                - olens (torch.Tensor): A tensor of shape (batch_size,)
                  representing the lengths of the outputs after processing.

        Examples:
            >>> encoder = ConformerPostEncoder(input_size=128)
            >>> input_tensor = torch.randn(32, 100, 128)  # Batch of 32
            >>> input_lengths = torch.tensor([100] * 32)  # All sequences have length 100
            >>> output, output_lengths = encoder(input_tensor, input_lengths)
            >>> output.shape
            torch.Size([32, 100, 256])  # Assuming output size is 256

        Note:
            The input tensor should be appropriately padded and the
            input_lengths should reflect the actual lengths of the
            sequences to ensure correct masking.

        Raises:
            ValueError: If the input tensor's shape does not match the
                expected dimensions or if the input_lengths tensor has
                an incompatible size.
        """
        xs_pad = input
        masks = (~make_pad_mask(input_lengths)).to(input[0].device)
        # print(mask)
        if self.embed is None:
            xs_pad = xs_pad
        else:
            xs_pad = self.embed(xs_pad)
        masks = masks.reshape(masks.shape[0], 1, masks.shape[1])
        xs_pad, masks = self.encoders(xs_pad, masks)

        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0]
        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        olens = masks.squeeze(1).sum(1)

        return xs_pad, olens

    def output_size(self) -> int:
        """
        Get the output size of the ConformerPostEncoder.

        This method returns the output size that was specified during the
        initialization of the ConformerPostEncoder. The output size is the
        dimension of the attention layer, which is used to shape the output
        of the encoder.

        Returns:
            int: The output size of the encoder.

        Examples:
            >>> conformer_post_encoder = ConformerPostEncoder(output_size=512)
            >>> conformer_post_encoder.output_size()
            512
        """
        return self._output_size
