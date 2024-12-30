"""Normalized position-wise feed-forward module for MEGA block."""

import torch


class NormalizedPositionwiseFeedForward(torch.nn.Module):
    """
    Normalized position-wise feed-forward module for MEGA block.

    This module implements a normalized position-wise feed-forward layer that is
    commonly used in transformer architectures. It applies linear transformations
    to the input data, followed by an activation function, dropout, and normalization.

    Attributes:
        linear1 (torch.nn.Linear): First linear transformation layer.
        linear2 (torch.nn.Linear): Second linear transformation layer.
        normalization (torch.nn.Module): Normalization module to apply.
        activation (torch.nn.Module): Activation function to apply.
        dropout (torch.nn.Dropout): Dropout layer for regularization.
        hidden_dropout (torch.nn.Dropout): Dropout layer for hidden units.

    Args:
        size (int): Input/Output size.
        hidden_size (int): Hidden size.
        normalization (torch.nn.Module, optional): Normalization module (default:
            torch.nn.LayerNorm).
        activation (torch.nn.Module, optional): Activation function (default:
            torch.nn.ReLU).
        dropout_rate (float, optional): Dropout rate (default: 0.0).

    Examples:
        >>> ff = NormalizedPositionwiseFeedForward(size=512, hidden_size=2048)
        >>> input_tensor = torch.randn(32, 10, 512)  # (B, L, size)
        >>> output_tensor = ff(input_tensor)
        >>> output_tensor.shape
        torch.Size([32, 10, 512])

    Note:
        The module is designed to be used within a larger neural network
        architecture, particularly in sequence-to-sequence tasks.

    Raises:
        ValueError: If `size` or `hidden_size` is non-positive.

    Todo:
        Add support for additional activation functions and normalization layers.
    """

    def __init__(
        self,
        size: int,
        hidden_size: int,
        normalization: torch.nn.Module = torch.nn.LayerNorm,
        activation: torch.nn.Module = torch.nn.ReLU,
        dropout_rate: float = 0.0,
    ) -> None:
        """Construct an NormalizedPositionwiseFeedForward object."""
        super().__init__()
        self.linear1 = torch.nn.Linear(size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, size)

        self.normalization = normalization
        self.activation = activation

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.hidden_dropout = torch.nn.Dropout(p=dropout_rate)

        self.reset_parameters()

    def reset_parameters(self, val: float = 0.0, std: float = 0.02) -> None:
        """
        Reset module parameters.

        This method initializes the weights and biases of the linear layers in
        the NormalizedPositionwiseFeedForward module. The weights are initialized
        using a normal distribution with a specified mean and standard deviation,
        while the biases are initialized to a constant value.

        Args:
            val: Initialization value for biases and the mean of the weight
                initialization (default is 0.0).
            std: Standard deviation for the normal distribution used to
                initialize the weights (default is 0.02).

        Examples:
            >>> ff = NormalizedPositionwiseFeedForward(size=512, hidden_size=2048)
            >>> ff.reset_parameters(val=0.1, std=0.01)

        Note:
            This method should be called if you want to reinitialize the
            parameters of the model after it has been created or modified.
        """
        torch.nn.init.normal_(self.linear1.weight, mean=val, std=std)
        torch.nn.init.constant_(self.linear1.bias, val)

        torch.nn.init.normal_(self.linear2.weight, mean=val, std=std)
        torch.nn.init.constant_(self.linear2.bias, val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute feed-forward module.

        This method applies a two-layer feed-forward network with
        residual connections and normalization. It first passes the input
        through a linear layer followed by an activation function,
        applies dropout, and then passes the result through a second
        linear layer. The input is added back to the output (residual
        connection), and finally, the result is normalized.

        Args:
            x: Input tensor of shape (B, L, size), where:
                B is the batch size,
                L is the sequence length, and
                size is the input/output size.

        Returns:
            torch.Tensor: Output tensor of the same shape as input (B, L, size).

        Examples:
            >>> ff = NormalizedPositionwiseFeedForward(size=512, hidden_size=2048)
            >>> input_tensor = torch.rand(32, 10, 512)  # Batch of 32, sequence length of 10
            >>> output_tensor = ff.forward(input_tensor)
            >>> print(output_tensor.shape)  # Should output: torch.Size([32, 10, 512])

        Note:
            The activation function and normalization module can be
            customized during the initialization of the
            NormalizedPositionwiseFeedForward object.
        """
        residual = x

        x = self.hidden_dropout(self.activation(self.linear1(x)))
        x = self.dropout(self.linear2(x))

        x = x + residual

        x = self.normalization(x)

        return x
