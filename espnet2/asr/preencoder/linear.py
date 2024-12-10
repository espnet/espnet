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

    This class implements a linear projection preencoder, which is used in
    automatic speech recognition (ASR) tasks. It transforms input features
    from one dimension to another using a linear layer followed by dropout
    for regularization. The output of this layer is suitable for further
    processing in ASR models.

    Attributes:
        output_dim (int): The dimension of the output features after
            projection.
        linear_out (torch.nn.Linear): The linear transformation layer that
            projects the input features to the output dimension.
        dropout (torch.nn.Dropout): The dropout layer applied to the input
            features for regularization.

    Args:
        input_size (int): The number of input features (dimension of input).
        output_size (int): The number of output features (dimension of output).
        dropout (float, optional): The dropout probability (default is 0.0).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - output (torch.Tensor): The projected output features.
            - input_lengths (torch.Tensor): The lengths of the input sequences
            (unchanged).

    Examples:
        >>> import torch
        >>> linear_projection = LinearProjection(input_size=128, output_size=64)
        >>> input_tensor = torch.rand(32, 10, 128)  # (batch_size, seq_len, input_size)
        >>> input_lengths = torch.tensor([10] * 32)  # All sequences are of length 10
        >>> output, lengths = linear_projection(input_tensor, input_lengths)
        >>> output.shape
        torch.Size([32, 10, 64])  # Output shape is (batch_size, seq_len, output_size)

    Note:
        This preencoder does not maintain any state across forward passes,
        which makes it suitable for use in stateless models.

    Todo:
        - Add support for different activation functions or additional layers
        if needed.
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
        Perform the forward pass of the LinearProjection module.

        This method applies a linear transformation followed by dropout to the
        input tensor. The output is a transformed tensor along with the original
        input lengths, which are unchanged.

        Args:
            input (torch.Tensor): The input tensor of shape (batch_size,
                input_size) containing the data to be processed.
            input_lengths (torch.Tensor): A tensor containing the lengths of
                each input sequence in the batch. This should have shape
                (batch_size,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - output (torch.Tensor): The output tensor after applying
                  linear transformation and dropout, of shape
                  (batch_size, output_size).
                - input_lengths (torch.Tensor): The unchanged input lengths
                  tensor.

        Examples:
            >>> model = LinearProjection(input_size=128, output_size=64, dropout=0.1)
            >>> input_tensor = torch.randn(32, 128)  # batch of 32
            >>> lengths = torch.tensor([128] * 32)  # all sequences have length 128
            >>> output, lengths = model.forward(input_tensor, lengths)
            >>> print(output.shape)  # should be (32, 64)
            >>> print(lengths.shape)  # should be (32,)

        Note:
            The dropout layer is applied during training mode only. If the model
            is in evaluation mode, dropout will not be applied.
        """
        output = self.linear_out(self.dropout(input))
        return output, input_lengths  # no state in this layer

    def output_size(self) -> int:
        """
        Returns the output size of the linear projection.

        This method retrieves the output dimension set during the initialization of
        the LinearProjection instance. The output size corresponds to the number of
        output features produced by the linear transformation applied in the
        forward pass.

        Returns:
            int: The output size of the linear projection.

        Examples:
            # Example usage
            lp = LinearProjection(input_size=128, output_size=64)
            output_size = lp.output_size()
            print(output_size)  # Output: 64

        Note:
            This method does not take any arguments and simply returns the
            pre-defined output dimension.
        """
        return self.output_dim
