import torch
import torch.nn as nn


class WN(torch.nn.Module):
    """
        WN is a WaveNet-like neural network module designed for generative tasks.

    This module implements a series of dilated convolutions, allowing for
    efficient processing of sequential data. It includes options for conditional
    input and dropout regularization. The architecture supports multiple layers
    with varying dilation rates, which helps capture long-range dependencies in
    the input data.

    Attributes:
        hidden_channels (int): The number of hidden channels in the network.
        kernel_size (tuple): The size of the convolution kernel.
        dilation_rate (int): The rate of dilation for the convolutions.
        n_layers (int): The number of convolutional layers in the network.
        gin_channels (int): The number of channels for conditional input (default 0).
        p_dropout (float): The dropout probability for regularization (default 0).
        in_layers (ModuleList): List of input convolutional layers.
        res_skip_layers (ModuleList): List of residual and skip connection layers.
        drop (Dropout): Dropout layer for regularization.
        cond_layer (Conv1d): Conditional layer for processing additional input.

    Args:
        hidden_channels (int): Number of hidden channels in the network.
        kernel_size (int): Size of the convolution kernel (must be odd).
        dilation_rate (int): Dilation rate for convolutions.
        n_layers (int): Number of layers in the network.
        gin_channels (int, optional): Number of input channels for conditional input.
            Defaults to 0 (no conditional input).
        p_dropout (float, optional): Probability of dropout. Defaults to 0.

    Returns:
        Tensor: The output tensor after processing through the network.

    Raises:
        AssertionError: If kernel_size is not odd.

    Examples:
        >>> model = WN(hidden_channels=64, kernel_size=3, dilation_rate=2,
        ...            n_layers=5, gin_channels=10, p_dropout=0.1)
        >>> x = torch.randn(1, 64, 100)  # Batch size 1, 64 channels, 100 length
        >>> x_mask = torch.ones(1, 1, 100)  # No masking
        >>> output = model(x, x_mask)

    Note:
        The input tensor `x` should have the shape (batch_size, hidden_channels,
        sequence_length). The mask tensor `x_mask` should have the shape
        (batch_size, 1, sequence_length) and is used to mask the output.

    Todo:
        - Add support for variable kernel sizes.
        - Implement a mechanism to handle variable input lengths.
    """

    def __init__(
        self,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
        p_dropout=0,
    ):
        super(WN, self).__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels > 0:
            cond_layer = torch.nn.Conv1d(
                gin_channels, 2 * hidden_channels * n_layers, 1
            )
            self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name="weight")

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            in_layer = torch.nn.utils.weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, g=None, **kwargs):
        """
            Performs the forward pass of the WN model.

        This method computes the output of the WN model given the input tensor `x`,
        an input mask `x_mask`, and an optional conditioning tensor `g`. It applies
        several layers of convolutions followed by non-linear activations and
        dropout. The output is computed as a weighted sum of the input and the
        skip connections from the convolutional layers.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size,
                hidden_channels, sequence_length).
            x_mask (torch.Tensor): A binary mask tensor of shape (batch_size,
                1, sequence_length) to control the contribution of each
                time step in the input.
            g (torch.Tensor, optional): An optional conditioning tensor of shape
                (batch_size, gin_channels, sequence_length). If provided, it is
                passed through a conditioning layer.
            **kwargs: Additional keyword arguments for future extension.

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, hidden_channels,
                sequence_length), after applying the WN model transformations.

        Examples:
            >>> model = WN(hidden_channels=64, kernel_size=3, dilation_rate=2,
            ...     n_layers=4)
            >>> x = torch.randn(10, 64, 50)  # Example input
            >>> x_mask = torch.ones(10, 1, 50)  # Example mask
            >>> output = model.forward(x, x_mask)
            >>> output.shape
            torch.Size([10, 64, 50])

        Note:
            This method utilizes weight normalization for the convolutional layers
            to stabilize training.

        Raises:
            ValueError: If the input tensor `x` does not match the expected shape.
        """
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = self.fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        """
            Remove weight normalization from the WN model layers.

        This method removes weight normalization from all layers of the WN model,
        including the conditional layer (if it exists), input layers, and
        residual-skip layers. This is typically used to revert the layers back
        to their original state after weight normalization has been applied.

        Attributes:
            gin_channels (int): Number of conditional input channels. If greater
                than 0, the conditional layer will also have weight normalization
                removed.

        Examples:
            >>> model = WN(hidden_channels=64, kernel_size=3, dilation_rate=2,
            ...            n_layers=4, gin_channels=0)
            >>> model.remove_weight_norm()  # Removes weight normalization from model

        Note:
            This method does not return any value but modifies the model's layers
            in place.

        Raises:
            ValueError: If the model has not been initialized properly.
        """
        if self.gin_channels != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        for in_layer in self.in_layers:
            torch.nn.utils.remove_weight_norm(in_layer)
        for res_skip_layer in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(res_skip_layer)

    def fused_add_tanh_sigmoid_multiply(self, input_a, input_b, n_channels):
        """
                Computes the fused operation of addition, tanh, sigmoid, and multiplication.

        This function takes two input tensors, adds them together, applies the tanh and
        sigmoid activation functions to the result, and finally multiplies the outputs
        of the tanh and sigmoid functions.

        Attributes:
            input_a (torch.Tensor): The first input tensor.
            input_b (torch.Tensor): The second input tensor.
            n_channels (torch.IntTensor): A tensor containing the number of channels.

        Args:
            input_a (torch.Tensor): The first input tensor of shape (batch_size,
                hidden_channels, sequence_length).
            input_b (torch.Tensor): The second input tensor of shape (batch_size,
                hidden_channels, sequence_length).
            n_channels (torch.IntTensor): A tensor containing the number of hidden
                channels as its first element.

        Returns:
            torch.Tensor: The result of the fused operation, with shape
                (batch_size, hidden_channels, sequence_length).

        Examples:
            >>> input_a = torch.randn(10, 16, 50)  # batch_size=10, hidden_channels=16
            >>> input_b = torch.randn(10, 16, 50)
            >>> n_channels = torch.IntTensor([16])
            >>> output = fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels)
            >>> output.shape
            torch.Size([10, 16, 50])

        Note:
            This function assumes that the first dimension of both input tensors is the
            batch size and the second dimension corresponds to the number of channels.

        Todo:
            - Add support for additional input types (e.g., numpy arrays) in the future.
        """
        n_channels_int = n_channels[0]
        in_act = input_a + input_b
        t_act = torch.tanh(in_act[:, :n_channels_int, :])
        s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
        acts = t_act * s_act
        return acts
