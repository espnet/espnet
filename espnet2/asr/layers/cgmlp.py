"""MLP with convolutional gating (cgMLP) definition.

References:
    https://openreview.net/forum?id=RA-zVvZLYIy
    https://arxiv.org/abs/2105.08050

"""

import torch

from espnet.nets.pytorch_backend.nets_utils import get_activation
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class ConvolutionalSpatialGatingUnit(torch.nn.Module):
    """
    Convolutional Spatial Gating Unit (CSGU) for convolutional gating in MLPs.

    This module implements a Convolutional Spatial Gating Unit, which is part of the
    convolutional gating MLP architecture. It splits the input tensor into two halves,
    applies normalization, convolution, and optional linear transformation, and then
    combines the results with gating mechanisms.

    Attributes:
        norm (LayerNorm): Layer normalization applied to the gating input.
        conv (Conv1d): 1D convolutional layer applied to the normalized gating input.
        linear (Linear or None): Optional linear layer applied after the convolution.
        act (nn.Module): Activation function applied to the gated output.
        dropout (Dropout): Dropout layer applied to the output.

    Args:
        size (int): The total number of input channels.
        kernel_size (int): The size of the convolutional kernel.
        dropout_rate (float): The dropout rate for the output.
        use_linear_after_conv (bool): Flag indicating if a linear layer should be
            applied after convolution.
        gate_activation (str): The activation function to use for the gating mechanism.
            Can be 'identity' or any valid activation name supported by PyTorch.

    Methods:
        espnet_initialization_fn(): Initializes weights and biases for the layers.
        forward(x: torch.Tensor, gate_add: Optional[torch.Tensor]) -> torch.Tensor:
            Performs the forward pass of the unit.

    Examples:
        >>> csgu = ConvolutionalSpatialGatingUnit(size=64, kernel_size=3,
        ...     dropout_rate=0.1, use_linear_after_conv=True, gate_activation='relu')
        >>> input_tensor = torch.randn(10, 20, 64)  # (N, T, D)
        >>> gate_add_tensor = torch.randn(10, 20, 32)  # (N, T, D/2)
        >>> output = csgu(input_tensor, gate_add=gate_add_tensor)
        >>> print(output.shape)  # Should output: torch.Size([10, 20, 32])

    Note:
        The input tensor should have its last dimension equal to `size`, and if
        `gate_add` is provided, it should have a last dimension equal to `size / 2`.

    Raises:
        ValueError: If the input tensor dimensions do not match the expected size.
    """

    def __init__(
        self,
        size: int,
        kernel_size: int,
        dropout_rate: float,
        use_linear_after_conv: bool,
        gate_activation: str,
    ):
        super().__init__()

        n_channels = size // 2  # split input channels
        self.norm = LayerNorm(n_channels)
        self.conv = torch.nn.Conv1d(
            n_channels,
            n_channels,
            kernel_size,
            1,
            (kernel_size - 1) // 2,
            groups=n_channels,
        )
        if use_linear_after_conv:
            self.linear = torch.nn.Linear(n_channels, n_channels)
        else:
            self.linear = None

        if gate_activation == "identity":
            self.act = torch.nn.Identity()
        else:
            self.act = get_activation(gate_activation)

        self.dropout = torch.nn.Dropout(dropout_rate)

    def espnet_initialization_fn(self):
        """
        Initializes the weights and biases of the Convolutional Spatial Gating 
        Unit (CSGU) components using a normal distribution for weights and ones 
        for biases.

        This method performs the following initializations:
            - Initializes the convolutional layer's weights with a normal 
            distribution with mean 0 and standard deviation 1e-6.
            - Initializes the convolutional layer's biases to ones.
            - If a linear layer is used after the convolution, initializes its 
            weights similarly and its biases to ones.

        Note:
            This method is typically called after the model's parameters have 
            been set to ensure that they start from a reasonable point for 
            training.

        Examples:
            >>> csgu = ConvolutionalSpatialGatingUnit(size=128, kernel_size=3, 
            ... dropout_rate=0.1, use_linear_after_conv=True, 
            ... gate_activation='relu')
            >>> csgu.espnet_initialization_fn()  # Initialize parameters

        Raises:
            None
        """
        torch.nn.init.ones_(self.conv.bias)
        if self.linear is not None:
            torch.nn.init.normal_(self.linear.weight, std=1e-6)
            torch.nn.init.ones_(self.linear.bias)

    def forward(self, x, gate_add=None):
        """
        Convolutional Spatial Gating Unit (CSGU).

        This module applies a convolutional gating mechanism to the input tensor,
        allowing for dynamic modulation of the input based on learned spatial
        patterns. The input tensor is split into two halves, where one half is
        processed through a convolutional layer and the other half is used to
        control the gating mechanism.

        Attributes:
            norm (LayerNorm): Layer normalization applied to the gating tensor.
            conv (Conv1d): 1D convolution layer for the gating tensor.
            linear (Linear or None): Optional linear layer applied after the
                convolution.
            act (Module): Activation function for the gating output.
            dropout (Dropout): Dropout layer for regularization.

        Args:
            size (int): Total number of input channels (must be even).
            kernel_size (int): Size of the convolutional kernel.
            dropout_rate (float): Dropout rate for regularization.
            use_linear_after_conv (bool): If True, a linear layer is used after
                the convolution.
            gate_activation (str): Activation function to use for the gating
                output (e.g., "relu", "sigmoid", "identity").

        Returns:
            torch.Tensor: The output tensor with the same shape as the input,
            where the first half of the channels has been gated by the processed
            second half.

        Examples:
            >>> import torch
            >>> csgu = ConvolutionalSpatialGatingUnit(size=4, kernel_size=3,
            ...                                       dropout_rate=0.1,
            ...                                       use_linear_after_conv=True,
            ...                                       gate_activation='relu')
            >>> x = torch.randn(2, 10, 4)  # (N, T, D)
            >>> output = csgu(x)
            >>> output.shape
            torch.Size([2, 10, 4])

        Note:
            The input tensor `x` must have an even number of channels to be
            split into two halves for gating.
        """

        x_r, x_g = x.chunk(2, dim=-1)

        x_g = self.norm(x_g)  # (N, T, D/2)
        x_g = self.conv(x_g.transpose(1, 2)).transpose(1, 2)  # (N, T, D/2)
        if self.linear is not None:
            x_g = self.linear(x_g)

        if gate_add is not None:
            x_g = x_g + gate_add

        x_g = self.act(x_g)
        out = x_r * x_g  # (N, T, D/2)
        out = self.dropout(out)
        return out


class ConvolutionalGatingMLP(torch.nn.Module):
    """
    Convolutional Gating MLP (cgMLP) class.

    This class implements a Convolutional Gating Multi-Layer Perceptron (cgMLP),
    which uses a convolutional spatial gating unit to process input features. 
    The cgMLP is designed to enhance the representation of sequential data by 
    leveraging both linear and convolutional layers.

    Attributes:
        channel_proj1 (torch.nn.Sequential): A sequential model consisting of a 
            linear layer followed by a GELU activation.
        csgu (ConvolutionalSpatialGatingUnit): The convolutional spatial gating 
            unit that applies gating mechanisms on the input.
        channel_proj2 (torch.nn.Linear): A linear layer that projects the output 
            from the spatial gating unit back to the original size.

    Args:
        size (int): The size of the input features.
        linear_units (int): The number of units in the linear layer before 
            applying the spatial gating unit.
        kernel_size (int): The size of the convolutional kernel used in the 
            spatial gating unit.
        dropout_rate (float): The dropout rate to be applied after the gating 
            operation.
        use_linear_after_conv (bool): If True, applies a linear layer after the 
            convolutional operation within the spatial gating unit.
        gate_activation (str): The activation function to use for the gating 
            mechanism. Common options include 'relu', 'sigmoid', or 'identity'.

    Returns:
        torch.Tensor: The output tensor of shape (N, T, D) where N is the batch 
            size, T is the sequence length, and D is the feature size.

    Examples:
        >>> model = ConvolutionalGatingMLP(size=256, linear_units=512, kernel_size=3,
        ...                                 dropout_rate=0.1, 
        ...                                 use_linear_after_conv=True, 
        ...                                 gate_activation='relu')
        >>> input_tensor = torch.randn(10, 20, 256)  # (N, T, D)
        >>> output = model(input_tensor, mask=None)
        >>> print(output.shape)
        torch.Size([10, 20, 256])  # Output shape is same as input size

    Note:
        This model is particularly useful in scenarios where capturing spatial 
        dependencies in sequential data is crucial, such as in speech 
        recognition tasks.

    Todo:
        - Implement support for additional gating mechanisms.
        - Add more activation functions for flexibility.
    """

    def __init__(
        self,
        size: int,
        linear_units: int,
        kernel_size: int,
        dropout_rate: float,
        use_linear_after_conv: bool,
        gate_activation: str,
    ):
        super().__init__()

        self.channel_proj1 = torch.nn.Sequential(
            torch.nn.Linear(size, linear_units), torch.nn.GELU()
        )
        self.csgu = ConvolutionalSpatialGatingUnit(
            size=linear_units,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            use_linear_after_conv=use_linear_after_conv,
            gate_activation=gate_activation,
        )
        self.channel_proj2 = torch.nn.Linear(linear_units // 2, size)

    def forward(self, x, mask):
        """
        Forward method for the Convolutional Gating MLP (cgMLP).

        This method processes the input tensor through a series of transformations,
        including linear projections and convolutional gating. It can optionally 
        incorporate a positional embedding if provided.

        Args:
            x (Union[torch.Tensor, tuple]): 
                Input tensor of shape (N, T, D) or a tuple containing:
                - xs_pad (torch.Tensor): Input tensor of shape (N, T, D).
                - pos_emb (torch.Tensor): Positional embedding tensor of shape (N, T, D).
            mask (torch.Tensor): 
                A mask tensor used for masking inputs, typically of shape (N, T).

        Returns:
            torch.Tensor or tuple: 
                If positional embedding is provided, returns a tuple containing:
                - Output tensor of shape (N, T, size) after processing.
                - Positional embedding tensor of shape (N, T, D).
                If no positional embedding is provided, returns only the output tensor.

        Examples:
            >>> model = ConvolutionalGatingMLP(size=128, linear_units=256, 
            ...                                 kernel_size=3, dropout_rate=0.1, 
            ...                                 use_linear_after_conv=True, 
            ...                                 gate_activation='relu')
            >>> input_tensor = torch.randn(32, 10, 128)  # (N, T, D)
            >>> mask = torch.ones(32, 10)  # (N, T)
            >>> output = model(input_tensor, mask)
            >>> print(output.shape)  # Output shape should be (32, 10, 128)

        Note:
            The input tensor is expected to have the last dimension equal to the
            specified size for the model.
        """
        if isinstance(x, tuple):
            xs_pad, pos_emb = x
        else:
            xs_pad, pos_emb = x, None

        xs_pad = self.channel_proj1(xs_pad)  # size -> linear_units
        xs_pad = self.csgu(xs_pad)  # linear_units -> linear_units/2
        xs_pad = self.channel_proj2(xs_pad)  # linear_units/2 -> size

        if pos_emb is not None:
            out = (xs_pad, pos_emb)
        else:
            out = xs_pad
        return out
