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
        Transformer encoder module for speech recognition tasks.

    This class implements a Transformer-based encoder that can be used in various
    speech recognition models. It supports different input layer types, positional
    encodings, and customizable encoder architectures.

    Attributes:
        _output_size (int): The output dimension of the encoder.
        embed (torch.nn.Module): The input embedding layer.
        normalize_before (bool): Whether to apply layer normalization before each block.
        encoders (torch.nn.Module): The main encoder layers.
        after_norm (torch.nn.Module): The final layer normalization (if normalize_before is True).
        num_inf (int): The number of inference outputs.
        encoders_sd (torch.nn.ModuleList): Speaker-dependent encoder layers.

    Args:
        input_size (int): Dimension of the input features.
        output_size (int, optional): Dimension of the encoder output and attention.
            Defaults to 256.
        attention_heads (int, optional): Number of attention heads. Defaults to 4.
        linear_units (int, optional): Number of units in position-wise feed-forward layers.
            Defaults to 2048.
        num_blocks (int, optional): Number of encoder blocks. Defaults to 6.
        num_blocks_sd (int, optional): Number of speaker-dependent encoder blocks.
            Defaults to 6.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
        positional_dropout_rate (float, optional): Dropout rate for positional encoding.
            Defaults to 0.1.
        attention_dropout_rate (float, optional): Dropout rate in attention layers.
            Defaults to 0.0.
        input_layer (str, optional): Type of input layer. Can be "conv2d", "linear",
            "conv2d1", "conv2d2", "conv2d6", "conv2d8", "embed", or None. Defaults to "conv2d".
        pos_enc_class (type, optional): Positional encoding class.
            Defaults to PositionalEncoding.
        normalize_before (bool, optional): Whether to use layer normalization before
            each block. Defaults to True.
        concat_after (bool, optional): Whether to concatenate attention layer's input
            and output. Defaults to False.
        positionwise_layer_type (str, optional): Type of position-wise layer.
            Can be "linear", "conv1d", or "conv1d-linear". Defaults to "linear".
        positionwise_conv_kernel_size (int, optional): Kernel size of position-wise
            conv1d layer. Defaults to 1.
        padding_idx (int, optional): Padding index for input_layer="embed". Defaults to -1.
        num_inf (int, optional): Number of inference outputs. Defaults to 1.

    Raises:
        ValueError: If an unknown input_layer type is specified.
        NotImplementedError: If an unsupported positionwise_layer_type is specified.

    Note:
        This implementation is based on the paper "Attention Is All You Need"
        by Vaswani et al. (2017) and adapted for speech recognition tasks.
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
                Get the output size of the encoder.

        Returns:
            int: The output dimension of the encoder.
        """
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
                Forward pass of the Transformer encoder.

        This method processes the input tensor through the encoder layers, applying
        positional encoding, attention mechanisms, and feed-forward networks.

        Args:
            xs_pad (torch.Tensor): Padded input tensor of shape (B, L, D), where B is
                the batch size, L is the sequence length, and D is the input dimension.
            ilens (torch.Tensor): Input lengths of each sequence in the batch, shape (B,).
            prev_states (torch.Tensor, optional): Not used in the current implementation.
                Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: A tuple containing:
                - torch.Tensor: Encoded output tensor of shape (B, num_inf, L', D'),
                  where L' is the encoded sequence length and D' is the output dimension.
                - torch.Tensor: Output lengths of each sequence in the batch after
                  encoding, shape (B, num_inf).
                - None: Placeholder for future use, currently always None.

        Raises:
            TooShortUttError: If the input sequence is too short for subsampling in
                certain input layer types (Conv2dSubsampling variants).

        Note:
            The method handles different input layer types and applies the appropriate
            embedding and subsampling techniques before passing the data through the
            encoder layers.
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
