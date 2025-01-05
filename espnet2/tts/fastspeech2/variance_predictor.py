#!/usr/bin/env python3

# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Variance predictor related modules."""

import torch
from typeguard import typechecked

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class VariancePredictor(torch.nn.Module):
    """
        Variance predictor module.

    This module implements the variance predictor described in `FastSpeech 2:
    Fast and High-Quality End-to-End Text to Speech`_.

    .. _`FastSpeech 2: Fast and High-Quality End-to-End Text to Speech`:
       https://arxiv.org/abs/2006.04558

    Attributes:
        conv (torch.nn.ModuleList): List of convolutional layers for variance
            prediction.
        linear (torch.nn.Linear): Linear layer for output prediction.

    Args:
        idim (int): Input dimension.
        n_layers (int): Number of convolutional layers.
        n_chans (int): Number of channels of convolutional layers.
        kernel_size (int): Kernel size of convolutional layers.
        bias (bool): Whether to use bias in convolutional layers.
        dropout_rate (float): Dropout rate.

    Examples:
        >>> vp = VariancePredictor(idim=256)
        >>> input_tensor = torch.rand(8, 100, 256)  # (B, Tmax, idim)
        >>> masks = torch.zeros(8, 100, dtype=torch.uint8)  # No padding
        >>> output = vp(input_tensor, masks)
        >>> print(output.shape)  # Output shape will be (8, 100, 1)

    Raises:
        TypeError: If any of the arguments are of the wrong type.

    Note:
        This module is designed for use in text-to-speech synthesis models.
    """

    @typechecked
    def __init__(
        self,
        idim: int,
        n_layers: int = 2,
        n_chans: int = 384,
        kernel_size: int = 3,
        bias: bool = True,
        dropout_rate: float = 0.5,
    ):
        """Initilize duration predictor module.

        Args:
            idim (int): Input dimension.
            n_layers (int): Number of convolutional layers.
            n_chans (int): Number of channels of convolutional layers.
            kernel_size (int): Kernel size of convolutional layers.
            dropout_rate (float): Dropout rate.

        """
        super().__init__()
        self.conv = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chans,
                        n_chans,
                        kernel_size,
                        stride=1,
                        padding=(kernel_size - 1) // 2,
                        bias=bias,
                    ),
                    torch.nn.ReLU(),
                    LayerNorm(n_chans, dim=1),
                    torch.nn.Dropout(dropout_rate),
                )
            ]
        self.linear = torch.nn.Linear(n_chans, 1)

    def forward(self, xs: torch.Tensor, x_masks: torch.Tensor = None) -> torch.Tensor:
        """
        Calculate forward propagation.

        This method processes the input sequences through the convolutional layers
        and returns the predicted variance for each sequence. It can handle padded
        inputs using the provided masks.

        Args:
            xs (Tensor): Batch of input sequences with shape (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded parts
                with shape (B, Tmax). Default is None.

        Returns:
            Tensor: Batch of predicted sequences with shape (B, Tmax, 1).

        Examples:
            >>> vp = VariancePredictor(idim=80)
            >>> input_tensor = torch.rand(32, 100, 80)  # (B, Tmax, idim)
            >>> mask_tensor = torch.zeros(32, 100, dtype=torch.bool)  # No padding
            >>> output = vp.forward(input_tensor, mask_tensor)
            >>> print(output.shape)  # Should print: torch.Size([32, 100, 1])

        Note:
            Ensure that the input tensor `xs` is appropriately shaped and
            the mask tensor, if provided, matches the dimensions of `xs`.
        """
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)

        xs = self.linear(xs.transpose(1, 2))  # (B, Tmax, 1)

        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0.0)

        return xs
