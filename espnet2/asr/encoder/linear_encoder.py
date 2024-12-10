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
    Linear encoder module for processing input features.

    This class implements a linear encoder that can use various input layer types,
    such as linear layers or convolutional subsampling layers. It applies an
    embedding operation to the input tensor and optionally normalizes the output
    before passing it to subsequent layers in a neural network.

    Attributes:
        _output_size (int): The dimension of the output features.
        embed (torch.nn.Module): The embedding layer, which can be a linear layer,
            convolutional subsampling, or embedding layer.
        normalize_before (bool): Flag indicating whether to apply layer normalization
            before the first block.
        after_norm (LayerNorm): Layer normalization applied to the output if
            normalize_before is True.

    Args:
        input_size (int): The dimension of the input features.
        output_size (int, optional): The dimension of the output features. Defaults
            to 256.
        dropout_rate (float, optional): The dropout rate for regularization. Defaults
            to 0.1.
        input_layer (str, optional): The type of input layer to use. Can be one of
            ['linear', 'conv2d', 'conv2d2', 'conv2d6', 'conv2d8', 'embed'].
            Defaults to 'conv2d'.
        normalize_before (bool, optional): Whether to apply layer normalization
            before the first block. Defaults to True.
        padding_idx (int, optional): The index for padding when using an embedding
            layer. Defaults to -1.

    Raises:
        ValueError: If an unknown input_layer type is specified.

    Examples:
        Initialize a linear encoder with a linear input layer:

            encoder = LinearEncoder(input_size=128, output_size=256,
                                    input_layer='linear')

        Forward pass through the encoder:

            xs_pad = torch.randn(10, 20, 128)  # (B, L, D)
            ilens = torch.tensor([20] * 10)     # Input lengths
            output, olens, _ = encoder(xs_pad, ilens)

    Note:
        The encoder expects the input tensor to be of shape (B, L, D) where B is
        the batch size, L is the sequence length, and D is the input dimension.
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
        Returns the output size of the linear encoder.

        This method provides the dimension of the output produced by the
        encoder, which is defined during the initialization of the
        LinearEncoder class.

        Returns:
            int: The output size of the encoder.

        Examples:
            encoder = LinearEncoder(input_size=128, output_size=256)
            size = encoder.output_size()  # size will be 256
        """
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Embed positions in tensor.

        This method processes the input tensor `xs_pad` by applying the
        embedding layer defined during the initialization of the LinearEncoder.
        It also handles padding masks based on the input lengths.

        Args:
            xs_pad (torch.Tensor): Input tensor of shape (B, L, D), where
                B is the batch size, L is the sequence length, and D is
                the dimension of the input features.
            ilens (torch.Tensor): A tensor containing the lengths of each
                input sequence in the batch. Shape (B).
            prev_states (torch.Tensor, optional): Not used currently. Defaults
                to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                A tuple containing:
                - The position-embedded tensor of shape (B, L, output_size).
                - A tensor representing the output lengths of shape (B).
                - An optional tensor which is currently set to None.

        Raises:
            TooShortUttError: If the input tensor is too short for the
                subsampling method being used.

        Examples:
            >>> encoder = LinearEncoder(input_size=128, output_size=256)
            >>> xs_pad = torch.randn(10, 20, 128)  # Batch of 10, 20 timesteps, 128 features
            >>> ilens = torch.tensor([20, 20, 20, 20, 20, 20, 20, 20, 20, 20])  # All lengths are 20
            >>> output, olens, _ = encoder.forward(xs_pad, ilens)
            >>> print(output.shape)  # Should be (10, 20, 256)
            >>> print(olens.shape)  # Should be (10,)
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
