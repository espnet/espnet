# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""
from typing import Optional, Tuple

import torch
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
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
from espnet.nets.pytorch_backend.transformer.subsampling import (
    Conv2dSubsampling,
    Conv2dSubsampling1,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    TooShortUttError,
    check_short_utt,
)


class TransformerEncoder(AbsEncoder):
    """
    Transformer encoder module for automatic speech recognition (ASR).

    This class implements a Transformer-based encoder that processes input
    features for ASR tasks. It supports multiple speaker-dependent encoder
    blocks and various input layer types, including linear and convolutional
    subsampling.

    Attributes:
        _output_size (int): The output dimension of the encoder.

    Args:
        input_size (int): Input dimension of the features.
        output_size (int, optional): Dimension of attention (default is 256).
        attention_heads (int, optional): Number of heads in multi-head attention
            (default is 4).
        linear_units (int, optional): Number of units in the position-wise feed
            forward network (default is 2048).
        num_blocks (int, optional): Number of recognition encoder blocks
            (default is 6).
        num_blocks_sd (int, optional): Number of speaker-dependent encoder blocks
            (default is 6).
        dropout_rate (float, optional): Dropout rate (default is 0.1).
        positional_dropout_rate (float, optional): Dropout rate after adding
            positional encoding (default is 0.1).
        attention_dropout_rate (float, optional): Dropout rate in attention
            (default is 0.0).
        input_layer (str, optional): Type of input layer (default is "conv2d").
        pos_enc_class: Class for positional encoding (default is
            PositionalEncoding).
        normalize_before (bool, optional): Whether to apply layer normalization
            before the first block (default is True).
        concat_after (bool, optional): Whether to concatenate input and output of
            attention layer (default is False).
        positionwise_layer_type (str, optional): Type of position-wise layer
            ("linear" or "conv1d", default is "linear").
        positionwise_conv_kernel_size (int, optional): Kernel size for
            position-wise conv1d layer (default is 1).
        padding_idx (int, optional): Padding index for input_layer="embed"
            (default is -1).
        num_inf (int, optional): Number of inference outputs (default is 1).

    Examples:
        >>> encoder = TransformerEncoder(input_size=80)
        >>> xs_pad = torch.randn(10, 50, 80)  # (B, L, D)
        >>> ilens = torch.tensor([50] * 10)  # Lengths of input sequences
        >>> output, olens, _ = encoder(xs_pad, ilens)

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            - Encoded output tensor of shape (B, num_inf, L, output_size).
            - Output lengths tensor of shape (B, num_inf).
            - Placeholder for future use (currently None).

    Raises:
        ValueError: If an unknown input layer type is provided.
        TooShortUttError: If the input tensor is too short for the selected
            subsampling method.

    Note:
        This encoder is designed for ASR tasks and may require additional
        components for full model integration.
    """

    @typechecked
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        num_blocks_sd: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        padding_idx: int = -1,
        num_inf: int = 1,
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
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d1":
            self.embed = Conv2dSubsampling1(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d2":
            self.embed = Conv2dSubsampling2(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(input_size, output_size, dropout_rate)
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(output_size, positional_dropout_rate)
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

        self.num_inf = num_inf
        self.encoders_sd = torch.nn.ModuleList(
            [
                repeat(
                    num_blocks_sd,
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
                for _ in range(num_inf)
            ]
        )

    def output_size(self) -> int:
        """
        output_size method.

        This method returns the output size of the TransformerEncoder, which is
        defined during the initialization of the encoder. The output size is the
        dimension of the attention mechanism used within the encoder.

        Returns:
            int: The output size of the TransformerEncoder.

        Examples:
            encoder = TransformerEncoder(input_size=128, output_size=256)
            print(encoder.output_size())  # Output: 256

        Note:
            The output size is crucial for ensuring that the dimensions match
            during the attention computations within the encoder layers.
        """
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Process input tensor through the transformer encoder.

        This method takes an input tensor and applies the transformer
        encoder layers, producing an output tensor that represents the
        encoded features of the input. The method also generates a mask
        to identify padded elements in the input tensor.

        Args:
            xs_pad: A tensor of shape (B, L, D) representing the input
                sequences, where B is the batch size, L is the sequence
                length, and D is the feature dimension.
            ilens: A tensor of shape (B) containing the lengths of the
                input sequences (without padding).
            prev_states: An optional tensor for previous states, currently
                not used in this implementation.

        Returns:
            A tuple containing:
                - A tensor of shape (B, num_inf, L, D) with the position
                  embedded tensor for each inference output.
                - A tensor of shape (B, num_inf) with the lengths of the
                  output sequences.
                - An optional tensor (currently None) for future use.

        Raises:
            TooShortUttError: If the input sequence is too short for the
                selected subsampling method, this error is raised.

        Examples:
            >>> encoder = TransformerEncoder(input_size=128)
            >>> xs_pad = torch.randn(10, 20, 128)  # Batch of 10 sequences
            >>> ilens = torch.tensor([20] * 10)     # All sequences have length 20
            >>> output, olens, _ = encoder.forward(xs_pad, ilens)

        Note:
            This method assumes that the input sequences have been properly
            preprocessed and padded. The encoder will not function correctly
            if the input tensor dimensions do not match the expected shapes.
        """
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

        if (
            isinstance(self.embed, Conv2dSubsampling)
            or isinstance(self.embed, Conv2dSubsampling1)
            or isinstance(self.embed, Conv2dSubsampling2)
            or isinstance(self.embed, Conv2dSubsampling6)
            or isinstance(self.embed, Conv2dSubsampling8)
        ):
            short_status, limit_size = check_short_utt(self.embed, xs_pad.size(1))
            if short_status:
                raise TooShortUttError(
                    f"has {xs_pad.size(1)} frames and is too short for subsampling "
                    + f"(it needs more than {limit_size} frames), return empty results",
                    xs_pad.size(1),
                    limit_size,
                )
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)

        xs_sd, masks_sd = [None] * self.num_inf, [None] * self.num_inf

        for ns in range(self.num_inf):
            xs_sd[ns], masks_sd[ns] = self.encoders_sd[ns](xs_pad, masks)
            xs_sd[ns], masks_sd[ns] = self.encoders(xs_sd[ns], masks_sd[ns])  # Enc_rec
            if self.normalize_before:
                xs_sd[ns] = self.after_norm(xs_sd[ns])

        olens = [masks_sd[ns].squeeze(1).sum(1) for ns in range(self.num_inf)]
        return torch.stack(xs_sd, dim=1), torch.stack(olens, dim=1), None
