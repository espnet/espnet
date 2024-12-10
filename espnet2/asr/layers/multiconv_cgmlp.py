"""Extension of convolutional gating (cgMLP) definition with multiple convolutions.

References:
    https://openreview.net/forum?id=RA-zVvZLYIy
    https://arxiv.org/abs/2105.08050
    https://arxiv.org/abs/2407.03718
"""

import torch

from espnet.nets.pytorch_backend.nets_utils import get_activation
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class MultiConvolutionalSpatialGatingUnit(torch.nn.Module):
    """
    Multi Convolutional Spatial Gating Unit (M-CSGU).

    This class implements a multi-convolutional spatial gating unit that
    applies several convolutional layers to input data and merges the outputs
    based on the specified architecture type. It can be used as a building
    block in advanced neural network architectures, particularly in
    applications involving speech and audio processing.

    Attributes:
        norm (LayerNorm): Layer normalization applied to the input channels.
        arch_type (str): Type of architecture for merging convolutions
            ('sum', 'weighted_sum', 'concat', or 'concat_fusion').
        convs (ModuleList): List of convolutional layers applied to the input.
        use_non_linear (bool): Flag indicating whether to apply a
            non-linear activation after convolution.
        kernel_prob_gen (Sequential): Module to generate kernel probabilities
            for weighted sum architecture.
        depthwise_conv_fusion (Conv1d): Convolution layer for concatenation
            fusion architecture.
        linear (Linear): Optional linear layer applied after convolutions.
        model_act: Activation function for the output.
        act: Activation function for gating.
        dropout (Dropout): Dropout layer for regularization.

    Args:
        size (int): Total number of input channels, which will be split in half.
        arch_type (str): Type of architecture for merging convolutions.
        kernel_sizes (str): Comma-separated string of kernel sizes for
            convolutional layers.
        merge_conv_kernel (int): Kernel size for the depthwise convolution
            in concatenation fusion.
        use_non_linear (bool): Whether to apply a non-linear activation
            after convolution.
        dropout_rate (float): Dropout rate for regularization.
        use_linear_after_conv (bool): Whether to apply a linear layer
            after convolutions.
        activation: Activation function to be used in the model.
        gate_activation (str): Activation function to be applied to the gating.

    Returns:
        out (torch.Tensor): Output tensor with shape (N, T, D/2).

    Raises:
        NotImplementedError: If an unknown architecture type is specified.

    Examples:
        >>> mcs_gating_unit = MultiConvolutionalSpatialGatingUnit(
        ...     size=64,
        ...     arch_type='sum',
        ...     kernel_sizes='3,5',
        ...     merge_conv_kernel=3,
        ...     use_non_linear=True,
        ...     dropout_rate=0.1,
        ...     use_linear_after_conv=True,
        ...     activation=torch.nn.ReLU(),
        ...     gate_activation='sigmoid'
        ... )
        >>> input_tensor = torch.randn(10, 20, 64)  # (N, T, D)
        >>> output_tensor = mcs_gating_unit(input_tensor)
        >>> output_tensor.shape
        torch.Size([10, 20, 32])  # Output shape is (N, T, D/2)

    Note:
        The input tensor should have an even number of channels, as it is
        split into two halves for processing.
    """

    def __init__(
        self,
        size: int,
        arch_type: str,
        kernel_sizes: str,
        merge_conv_kernel: int,
        use_non_linear: bool,
        dropout_rate: float,
        use_linear_after_conv: bool,
        activation,
        gate_activation: str,
    ):
        super().__init__()

        n_channels = size // 2  # split input channels
        self.norm = LayerNorm(n_channels)

        kernel_sizes = list(map(int, kernel_sizes.split(",")))
        no_kernels = len(kernel_sizes)

        assert (
            n_channels % no_kernels == 0
        ), f"{n_channels} input channels cannot be divided between {no_kernels} kernels"

        self.arch_type = arch_type
        if arch_type in ["sum", "weighted_sum"]:
            self.convs = torch.nn.ModuleList(
                [
                    torch.nn.Conv1d(
                        n_channels,
                        n_channels,
                        kernel_size,
                        1,
                        (kernel_size - 1) // 2,
                        groups=n_channels,
                    )
                    for kernel_size in kernel_sizes
                ]
            )
        elif arch_type in ["concat", "concat_fusion"]:
            self.convs = torch.nn.ModuleList(
                [
                    torch.nn.Conv1d(
                        n_channels,
                        n_channels // no_kernels,
                        kernel_size,
                        1,
                        (kernel_size - 1) // 2,
                        groups=n_channels // no_kernels,
                    )
                    for kernel_size in kernel_sizes
                ]
            )
        else:
            raise NotImplementedError(
                f"Unknown architecture type for MultiConvCGMLP: {arch_type}"
            )
        self.use_non_linear = use_non_linear
        if arch_type == "weighted_sum":
            self.kernel_prob_gen = torch.nn.Sequential(
                torch.nn.Linear(n_channels * no_kernels, no_kernels),
                torch.nn.Softmax(dim=-1),
            )
            self.depthwise_conv_fusion = None
        elif arch_type == "concat_fusion":
            self.kernel_prob_gen = None
            self.depthwise_conv_fusion = torch.nn.Conv1d(
                n_channels,
                n_channels,
                kernel_size=merge_conv_kernel,
                stride=1,
                padding=(merge_conv_kernel - 1) // 2,
                groups=n_channels,
                bias=True,
            )
        else:
            self.kernel_prob_gen = None
            self.depthwise_conv_fusion = None

        if use_linear_after_conv:
            self.linear = torch.nn.Linear(n_channels, n_channels)
        else:
            self.linear = None

        self.model_act = activation
        if gate_activation == "identity":
            self.act = torch.nn.Identity()
        else:
            self.act = get_activation(gate_activation)

        self.dropout = torch.nn.Dropout(dropout_rate)

    def espnet_initialization_fn(self):
        """
        Initializes the weights and biases of the convolutional layers.

        This method applies a normal distribution initialization to the weights
        of each convolutional layer and sets the biases to ones. It also handles
        the initialization for any additional linear layers present in the
        MultiConvolutionalSpatialGatingUnit.

        The initialization strategy used is as follows:
            - Convolutional weights are initialized using a normal distribution
            with a standard deviation of 1e-6.
            - Biases are initialized to one.

        This method is typically called after the model has been created to
        ensure that the parameters are initialized appropriately before training.

        Examples:
            >>> mcs_gu = MultiConvolutionalSpatialGatingUnit(size=64,
            ...     arch_type='sum', kernel_sizes='3,5', merge_conv_kernel=3,
            ...     use_non_linear=True, dropout_rate=0.1,
            ...     use_linear_after_conv=True, activation=torch.nn.ReLU(),
            ...     gate_activation='sigmoid')
            >>> mcs_gu.espnet_initialization_fn()  # Initialize parameters

        Note:
            Ensure that this method is called after creating an instance of the
            MultiConvolutionalSpatialGatingUnit and before starting training.

        Raises:
            NotImplementedError: If the architecture type does not match any
            known configuration.
        """
        for conv in self.convs:
            torch.nn.init.normal_(conv.weight, std=1e-6)
            torch.nn.init.ones_(conv.bias)
        if self.depthwise_conv_fusion is not None:
            torch.nn.init.normal_(self.depthwise_conv_fusion.weight, std=1e-6)
            torch.nn.init.ones_(self.depthwise_conv_fusion.bias)
        if self.linear is not None:
            torch.nn.init.normal_(self.linear.weight, std=1e-6)
            torch.nn.init.ones_(self.linear.bias)

    def forward(self, x, gate_add=None):
        """
        Perform the forward pass of the MultiConvolutionalSpatialGatingUnit.

        This method takes an input tensor and processes it through multiple
        convolutional layers, applying spatial gating mechanisms based on the
        specified architecture type. The output is then modulated by a gating
        mechanism, which can include an optional additive gate.

        Args:
            x (torch.Tensor): Input tensor of shape (N, T, D), where N is the
                batch size, T is the sequence length, and D is the number of
                input channels.
            gate_add (torch.Tensor, optional): Tensor of shape (N, T, D/2) to
                be added to the output after gating. If None, no addition is
                performed.

        Returns:
            out (torch.Tensor): Output tensor of shape (N, T, D/2) after
                applying convolutions, gating, and dropout.

        Examples:
            >>> model = MultiConvolutionalSpatialGatingUnit(size=64, arch_type='sum',
            ...     kernel_sizes='3,5', merge_conv_kernel=3, use_non_linear=True,
            ...     dropout_rate=0.1, use_linear_after_conv=True,
            ...     activation=torch.nn.ReLU(), gate_activation='sigmoid')
            >>> x = torch.randn(32, 10, 64)  # Example input
            >>> output = model.forward(x)
            >>> print(output.shape)  # Should output: torch.Size([32, 10, 32])

        Note:
            The input tensor is expected to be split into real and imaginary
            components, with the imaginary part undergoing normalization and
            convolution operations. The output tensor represents the gated
            product of the real component and the processed imaginary component.

        Todo:
            - Parallelize the convolution computation for improved performance.
        """
        x_r, x_i = x.chunk(2, dim=-1)

        x_i = self.norm(x_i).transpose(1, 2)  # (N, D/2, T)

        # TODO: Parallelize this convolution computation
        xs = []
        for conv in self.convs:
            xi = conv(x_i).transpose(1, 2)  # (N, T, D/2)
            if self.arch_type == "sum" and self.use_non_linear:
                xi = self.model_act(xi)
            xs.append(xi)

        if self.arch_type in ["sum", "weighted_sum"]:
            x = torch.stack(xs, dim=-2)
            if self.arch_type == "weighted_sum":
                prob = self.kernel_prob_gen(torch.cat(xs, dim=-1))
                x = prob.unsqueeze(-1) * x

            x_g = x.sum(dim=-2)
        else:
            x_concat = torch.cat(xs, dim=-1)  # (N, T, D)

            if self.arch_type == "concat_fusion":
                x_tmp = x_concat.transpose(1, 2)
                x_tmp = self.depthwise_conv_fusion(x_tmp)
                x_concat = x_concat + x_tmp.transpose(1, 2)

            x_g = x_concat

        if self.linear is not None:
            x_g = self.linear(x_g)

        if gate_add is not None:
            x_g = x_g + gate_add

        x_g = self.act(x_g)
        out = x_r * x_g  # (N, T, D/2)
        out = self.dropout(out)
        return out


class MultiConvolutionalGatingMLP(torch.nn.Module):
    """
    Convolutional Gating MLP (cgMLP).

    This class implements a multi-convolutional gating mechanism for MLPs,
    extending the capabilities of traditional MLPs with convolutional gating
    units (CSGUs). It can utilize various architectures to combine features
    from multiple convolutions and apply gating mechanisms for enhanced
    performance in tasks such as speech recognition.

    Attributes:
        channel_proj1 (torch.nn.Sequential): A sequential layer projecting
            input features to a higher-dimensional space with a GELU activation.
        csgu (MultiConvolutionalSpatialGatingUnit): An instance of the
            MultiConvolutionalSpatialGatingUnit class that performs the
            convolutional gating.
        channel_proj2 (torch.nn.Linear): A linear layer that projects the
            output back to the original size.

    Args:
        size (int): The dimensionality of the input features.
        linear_units (int): The number of linear units in the first projection.
        arch_type (str): The architecture type for the convolutional gating,
            can be 'sum', 'weighted_sum', 'concat', or 'concat_fusion'.
        kernel_sizes (str): A comma-separated string specifying the sizes of
            the convolutional kernels to be used.
        merge_conv_kernel (int): The kernel size for merging convolutions,
            applicable in 'concat_fusion' architecture.
        use_non_linear (bool): Whether to apply non-linear activation after
            convolution.
        dropout_rate (float): The dropout rate to be applied.
        use_linear_after_conv (bool): Whether to apply a linear layer after
            the convolutional layers.
        activation: The activation function to use for the model.
        gate_activation (str): The activation function to use for gating;
            typically 'identity' or other activation functions.

    Returns:
        torch.Tensor: The output tensor with the same dimensionality as the
            input features after passing through the MLP.

    Examples:
        >>> model = MultiConvolutionalGatingMLP(
        ...     size=256,
        ...     linear_units=512,
        ...     arch_type='sum',
        ...     kernel_sizes='3,5',
        ...     merge_conv_kernel=3,
        ...     use_non_linear=True,
        ...     dropout_rate=0.1,
        ...     use_linear_after_conv=True,
        ...     activation=torch.nn.GELU(),
        ...     gate_activation='identity'
        ... )
        >>> input_tensor = torch.randn(10, 256)  # Batch of 10 samples
        >>> output = model(input_tensor)
        >>> output.shape
        torch.Size([10, 256])

    Note:
        The implementation assumes that the input tensor has the correct
        dimensions and types. Ensure that the input tensor shape is
        compatible with the expected input size.

    Raises:
        NotImplementedError: If an unsupported architecture type is provided.
    """

    def __init__(
        self,
        size: int,
        linear_units: int,
        arch_type: str,
        kernel_sizes: str,
        merge_conv_kernel: int,
        use_non_linear: bool,
        dropout_rate: float,
        use_linear_after_conv: bool,
        activation,
        gate_activation: str,
    ):
        super().__init__()

        if arch_type not in ["sum", "weighted_sum", "concat", "concat_fusion"]:
            raise NotImplementedError(f"Unknown MultiConvCGMLP type: {type}")

        self.channel_proj1 = torch.nn.Sequential(
            torch.nn.Linear(size, linear_units), torch.nn.GELU()
        )
        self.csgu = MultiConvolutionalSpatialGatingUnit(
            size=linear_units,
            arch_type=arch_type,
            kernel_sizes=kernel_sizes,
            merge_conv_kernel=merge_conv_kernel,
            use_non_linear=use_non_linear,
            dropout_rate=dropout_rate,
            use_linear_after_conv=use_linear_after_conv,
            activation=activation,
            gate_activation=gate_activation,
        )
        self.channel_proj2 = torch.nn.Linear(linear_units // 2, size)

    def forward(self, x, mask=None):
        """
        Forward pass for the MultiConvolutionalGatingMLP.

        This method computes the output of the MultiConvolutionalGatingMLP
        by applying a series of linear transformations followed by the
        MultiConvolutionalSpatialGatingUnit (M-CSGU). The input is processed
        through the channel projection layers and the spatial gating unit to
        produce the final output.

        Args:
            x (Union[torch.Tensor, tuple]): Input tensor of shape (N, T, D) or
                a tuple containing the input tensor and positional embedding.
            mask (torch.Tensor, optional): An optional mask tensor to apply
                attention. Default is None.

        Returns:
            torch.Tensor or tuple: The output tensor of shape (N, T, D) or a
            tuple containing the output tensor and positional embedding if
            provided.

        Examples:
            >>> model = MultiConvolutionalGatingMLP(size=128, linear_units=256,
            ... arch_type='sum', kernel_sizes='3,5', merge_conv_kernel=3,
            ... use_non_linear=True, dropout_rate=0.1, use_linear_after_conv=True,
            ... activation=torch.nn.GELU(), gate_activation='relu')
            >>> input_tensor = torch.randn(32, 10, 128)  # (N, T, D)
            >>> output = model(input_tensor)
            >>> print(output.shape)  # Output: (32, 10, 128)

        Note:
            The input tensor `x` should have a shape compatible with the
            specified size parameter during initialization.
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
