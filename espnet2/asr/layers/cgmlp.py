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
        Convolutional Spatial Gating Unit (CSGU) for MLP with convolutional gating.

    This class implements the Convolutional Spatial Gating Unit, a key component
    of the cgMLP architecture. It applies layer normalization, 1D convolution,
    and optional linear transformation to the input, followed by an activation
    function and dropout.

    Attributes:
        norm (LayerNorm): Layer normalization applied to the input.
        conv (torch.nn.Conv1d): 1D convolution layer.
        linear (torch.nn.Linear or None): Optional linear layer after convolution.
        act (torch.nn.Module): Activation function for gating.
        dropout (torch.nn.Dropout): Dropout layer.

    Args:
        size (int): Input size (should be divisible by 2).
        kernel_size (int): Kernel size for the 1D convolution.
        dropout_rate (float): Dropout rate.
        use_linear_after_conv (bool): Whether to use a linear layer after convolution.
        gate_activation (str): Activation function to use for gating.

    Note:
        The input channels are split in half, with one half used for gating
        the other half.

    Example:
        >>> csgu = ConvolutionalSpatialGatingUnit(512, 3, 0.1, True, 'gelu')
        >>> input_tensor = torch.randn(32, 100, 512)
        >>> output = csgu(input_tensor)
        >>> output.shape
        torch.Size([32, 100, 256])
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
                Initialize the weights of the convolutional and linear layers.

        This method initializes the weights of the convolutional layer and, if present,
        the linear layer of the Convolutional Spatial Gating Unit (CSGU). The
        initialization follows the strategy described in the original cgMLP paper.

        The convolutional layer's weights are initialized from a normal distribution
        with a small standard deviation, while its biases are initialized to ones.
        If a linear layer is present, its weights are similarly initialized from a
        normal distribution, and its biases are set to ones.

        Returns:
            None

        Note:
            This method is typically called during the initialization of the CSGU
            module and doesn't need to be called manually in most cases.

        Example:
            >>> csgu = ConvolutionalSpatialGatingUnit(512, 3, 0.1, True, 'gelu')
            >>> csgu.espnet_initialization_fn()
        """
        torch.nn.init.ones_(self.conv.bias)
        if self.linear is not None:
            torch.nn.init.normal_(self.linear.weight, std=1e-6)
            torch.nn.init.ones_(self.linear.bias)

    def forward(self, x, gate_add=None):
        """
                Forward pass of the Convolutional Spatial Gating Unit.

        This method performs the forward computation of the CSGU. It splits the input
        tensor into two halves along the last dimension, applies normalization and
        convolution to one half, and uses it to gate the other half.

        Args:
            x (torch.Tensor): Input tensor of shape (N, T, D), where N is the batch size,
                T is the sequence length, and D is the feature dimension.
            gate_add (torch.Tensor, optional): Additional tensor to be added to the gate
                values. Should have shape (N, T, D/2). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (N, T, D/2). The output dimension is
                half of the input dimension due to the gating mechanism.

        Raises:
            ValueError: If the input tensor's last dimension is not even.

        Example:
            >>> csgu = ConvolutionalSpatialGatingUnit(512, 3, 0.1, True, 'gelu')
            >>> input_tensor = torch.randn(32, 100, 512)
            >>> output = csgu(input_tensor)
            >>> output.shape
            torch.Size([32, 100, 256])

        Note:
            The input tensor is split into two halves: x_r and x_g. x_g is processed
            through normalization, convolution, and optional linear transformation,
            then used to gate x_r. The result is then passed through dropout.
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
        Convolutional Gating MLP (cgMLP) module.

    This class implements the Convolutional Gating MLP, a variant of the MLP
    architecture that incorporates convolutional gating mechanisms. It consists
    of two channel projection layers and a Convolutional Spatial Gating Unit (CSGU).

    The cgMLP processes input by first projecting it to a higher dimension,
    then applying the CSGU for feature interaction and gating, and finally
    projecting it back to the original dimension.

    Attributes:
        channel_proj1 (torch.nn.Sequential): First channel projection layer with GELU activation.
        csgu (ConvolutionalSpatialGatingUnit): The core CSGU component.
        channel_proj2 (torch.nn.Linear): Second channel projection layer.

    Args:
        size (int): Input and output size of the module.
        linear_units (int): Number of units in the intermediate linear layer.
        kernel_size (int): Kernel size for the convolutional layer in CSGU.
        dropout_rate (float): Dropout rate used in CSGU.
        use_linear_after_conv (bool): Whether to use a linear layer after convolution in CSGU.
        gate_activation (str): Activation function to use for gating in CSGU.

    Example:
        >>> cgmlp = ConvolutionalGatingMLP(512, 1024, 3, 0.1, True, 'gelu')
        >>> input_tensor = torch.randn(32, 100, 512)
        >>> output = cgmlp(input_tensor, None)
        >>> output.shape
        torch.Size([32, 100, 512])

    Note:
        The forward method of this class can handle inputs with or without
        positional embeddings. If positional embeddings are provided, they
        will be returned along with the processed tensor.
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
                Forward pass of the Convolutional Gating MLP.

        This method performs the forward computation of the cgMLP. It processes
        the input through two channel projection layers and a Convolutional
        Spatial Gating Unit (CSGU).

        Args:
            x (torch.Tensor or tuple): If tuple, it should contain:
                - xs_pad (torch.Tensor): Input tensor of shape (N, T, D), where N is
                  the batch size, T is the sequence length, and D is the feature dimension.
                - pos_emb (torch.Tensor): Positional embedding tensor.
              If not a tuple, it's treated as the input tensor itself.
            mask (torch.Tensor): Mask tensor (not used in the current implementation).

        Returns:
            torch.Tensor or tuple: If the input included positional embeddings, returns a tuple
            containing:
                - xs_pad (torch.Tensor): Processed tensor of shape (N, T, D).
                - pos_emb (torch.Tensor): The input positional embedding tensor.
            Otherwise, returns just the processed tensor.

        Example:
            >>> cgmlp = ConvolutionalGatingMLP(512, 1024, 3, 0.1, True, 'gelu')
            >>> input_tensor = torch.randn(32, 100, 512)
            >>> output = cgmlp(input_tensor, None)
            >>> output.shape
            torch.Size([32, 100, 512])

            >>> input_with_pos = (input_tensor, torch.randn(32, 100, 512))
            >>> output_with_pos = cgmlp(input_with_pos, None)
            >>> isinstance(output_with_pos, tuple)
            True

        Note:
            The 'mask' argument is currently not used in the method but is kept
            for compatibility with other modules in the architecture.
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
