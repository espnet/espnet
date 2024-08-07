#!/usr/bin/env python3
#  2021, Carnegie Mellon University;  Xuankai Chang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Linear Projection."""

from typing import Tuple

import torch
from typeguard import typechecked

from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder


class LinearProjection(AbsPreEncoder):
    """
        Linear Projection Preencoder.

    This class implements a linear projection preencoder that applies a linear transformation
    followed by dropout to the input features. It is a subclass of AbsPreEncoder.

    Attributes:
        output_dim (int): The dimension of the output features.
        linear_out (torch.nn.Linear): The linear transformation layer.
        dropout (torch.nn.Dropout): The dropout layer.

    Args:
        input_size (int): The size of the input features.
        output_size (int): The size of the output features.
        dropout (float, optional): The dropout rate. Defaults to 0.0.

    Example:
        >>> import torch
        >>> preencoder = LinearProjection(input_size=100, output_size=80, dropout=0.1)
        >>> input_tensor = torch.randn(32, 10, 100)  # (batch_size, sequence_length, input_size)
        >>> input_lengths = torch.full((32,), 10)
        >>> output, output_lengths = preencoder(input_tensor, input_lengths)
        >>> print(output.shape)
        torch.Size([32, 10, 80])

    Note:
        This preencoder does not modify the input lengths, so the output_lengths
        will be the same as the input_lengths.
    """

    @typechecked
    def __init__(self, input_size: int, output_size: int, dropout: float = 0.0):
        """Initialize the module."""
        super().__init__()

        self.output_dim = output_size
        self.linear_out = torch.nn.Linear(input_size, output_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
                Forward pass of the LinearProjection preencoder.

        This method applies dropout to the input, then performs a linear transformation.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).
            input_lengths (torch.Tensor): Tensor of input sequence lengths of shape (batch_size,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - output (torch.Tensor): The transformed output tensor of shape
                  (batch_size, sequence_length, output_size).
                - input_lengths (torch.Tensor): The input sequence lengths, unchanged.

        Note:
            This method does not modify the input lengths, so the returned input_lengths
            are the same as the input.

        Example:
            >>> preencoder = LinearProjection(input_size=100, output_size=80)
            >>> input_tensor = torch.randn(32, 10, 100)
            >>> input_lengths = torch.full((32,), 10)
            >>> output, output_lengths = preencoder.forward(input_tensor, input_lengths)
            >>> print(output.shape)
            torch.Size([32, 10, 80])
            >>> print(torch.all(output_lengths == input_lengths))
            True
        """
        output = self.linear_out(self.dropout(input))
        return output, input_lengths  # no state in this layer

    def output_size(self) -> int:
        """
                Get the output size of the LinearProjection preencoder.

        Returns:
            int: The dimension of the output features.

        Example:
            >>> preencoder = LinearProjection(input_size=100, output_size=80)
            >>> print(preencoder.output_size())
            80

        Note:
            This method returns the value of the `output_dim` attribute, which is set
            during the initialization of the LinearProjection instance.
        """
        return self.output_dim
