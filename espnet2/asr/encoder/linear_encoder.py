# Copyright 2023 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Linear encoder definition."""

from typing import Optional, Tuple

import torch
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.subsampling import (
    Conv2dSubsampling,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    TooShortUttError,
    check_short_utt,
)


class LinearEncoder(AbsEncoder):
    """
        Linear encoder module for processing input sequences.

    This class implements a linear encoder that can be used in various sequence processing tasks,
    such as speech recognition or natural language processing. It supports different types of
    input layers and can apply normalization and dropout to the input.

    Attributes:
        embed: The input embedding layer, which can be one of several types (linear, conv2d, etc.).
        normalize_before: A boolean indicating whether to apply layer normalization before processing.
        after_norm: A LayerNorm layer applied after processing if normalize_before is True.

    Args:
        input_size (int): The dimensionality of the input features.
        output_size (int, optional): The dimensionality of the output features. Defaults to 256.
        dropout_rate (float, optional): The dropout rate to apply. Defaults to 0.1.
        input_layer (str, optional): The type of input layer to use. Can be 'linear', 'conv2d',
            'conv2d2', 'conv2d6', 'conv2d8', 'embed', or None. Defaults to 'conv2d'.
        normalize_before (bool, optional): Whether to apply layer normalization before processing.
            Defaults to True.
        padding_idx (int, optional): The index used for padding in the embedding layer.
            Only used when input_layer is 'embed'. Defaults to -1.

    Raises:
        ValueError: If an unknown input_layer type is specified.

    Examples:
        >>> encoder = LinearEncoder(input_size=80, output_size=256, input_layer='conv2d')
        >>> input_tensor = torch.randn(32, 1000, 80)  # (batch_size, time_steps, features)
        >>> input_lengths = torch.full((32,), 1000)
        >>> output, output_lengths, _ = encoder(input_tensor, input_lengths)
        >>> print(output.shape)
        torch.Size([32, 250, 256])  # Time dimension is reduced due to conv2d subsampling

    Note:
        The actual behavior of the encoder depends on the chosen input_layer type.
        Some input layers (like conv2d variants) may modify the sequence length.
    """

    @typechecked
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        dropout_rate: float = 0.1,
        input_layer: Optional[str] = "conv2d",
        normalize_before: bool = True,
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
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d2":
            self.embed = Conv2dSubsampling2(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(input_size, output_size, dropout_rate)
        elif input_layer == "embed":
            self.embed = (
                torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx),
            )
        elif input_layer is None:
            if input_size == output_size:
                self.embed = None
            else:
                self.embed = torch.nn.Linear(input_size, output_size)
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

    def output_size(self) -> int:
        """
                Get the output size of the encoder.

        Returns:
            int: The dimensionality of the output features.

        Example:
            >>> encoder = LinearEncoder(input_size=80, output_size=256)
            >>> print(encoder.output_size())
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
                Forward pass of the LinearEncoder.

        This method processes the input tensor through the encoder, applying the specified
        input layer, normalization, and any other transformations defined in the encoder.

        Args:
            xs_pad (torch.Tensor): Input tensor of shape (B, L, D), where B is the batch size,
                L is the sequence length, and D is the input feature dimension.
            ilens (torch.Tensor): Input lengths of each sequence in the batch, shape (B,).
            prev_states (torch.Tensor, optional): Not used in this implementation. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: A tuple containing:
                - xs_pad (torch.Tensor): Encoded output tensor of shape (B, L', D'),
                  where L' is the potentially modified sequence length and D' is the output dimension.
                - olens (torch.Tensor): Output lengths of each sequence in the batch, shape (B,).
                - None: Placeholder for consistency with other encoder implementations.

        Raises:
            TooShortUttError: If the input sequence is too short for subsampling operations
                when using certain input layers (e.g., Conv2dSubsampling variants).

        Examples:
            >>> encoder = LinearEncoder(input_size=80, output_size=256)
            >>> xs_pad = torch.randn(32, 1000, 80)  # (batch_size, time_steps, features)
            >>> ilens = torch.full((32,), 1000)
            >>> output, olens, _ = encoder(xs_pad, ilens)
            >>> print(output.shape, olens.shape)
            torch.Size([32, 1000, 256]) torch.Size([32])

        Note:
            The actual output shape may vary depending on the input_layer type used in the encoder.
            Some input layers (like conv2d variants) may reduce the sequence length.
        """
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

        if self.embed is None:
            xs_pad = xs_pad
        elif (
            isinstance(self.embed, Conv2dSubsampling)
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

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        olens = masks.squeeze(1).sum(1)
        return xs_pad, olens, None
