import torch
import torch.nn as nn
import torch.nn.functional as F


class NavieComplexLSTM(nn.Module):
    """
    A naive implementation of a complex-valued Long Short-Term Memory (LSTM).

    This LSTM processes complex-valued input by separating the real and
    imaginary parts and passing them through individual LSTMs. The outputs
    from these LSTMs are combined to produce the final complex-valued output.

    Attributes:
        bidirectional (bool): If True, the LSTM will be bidirectional.
        input_dim (int): The input dimension for the LSTM, half of the input size.
        rnn_units (int): The number of hidden units for the LSTM, half of the
            hidden size.
        real_lstm (nn.LSTM): LSTM for processing the real part of the input.
        imag_lstm (nn.LSTM): LSTM for processing the imaginary part of the input.
        projection_dim (int or None): Dimension for the output projection layer.
        r_trans (nn.Linear or None): Linear transformation for the real output.
        i_trans (nn.Linear or None): Linear transformation for the imaginary output.

    Args:
        input_size (int): The size of the input (must be even for complex inputs).
        hidden_size (int): The number of hidden units in the LSTM (must be even).
        projection_dim (int or None): The dimension of the projection layer
            (if None, no projection is applied).
        bidirectional (bool): If True, the LSTM will be bidirectional.
        batch_first (bool): If True, input and output tensors are provided
            as (batch, seq, feature).

    Returns:
        list: A list containing the real and imaginary outputs.

    Yields:
        None

    Raises:
        ValueError: If the input size is not even or if the hidden size is not even.

    Examples:
        >>> lstm = NavieComplexLSTM(input_size=4, hidden_size=4)
        >>> inputs = [torch.randn(10, 2), torch.randn(10, 2)]  # 10 time steps, 2 features
        >>> outputs = lstm(inputs)
        >>> real_out, imag_out = outputs
        >>> real_out.shape, imag_out.shape
        (torch.Size([10, 2]), torch.Size([10, 2]))

    Note:
        The inputs must be in the form of a list or a single tensor that can
        be split into real and imaginary components.

    Todo:
        Implement support for multi-layer LSTMs and dropout.
    """
    def __init__(
        self,
        input_size,
        hidden_size,
        projection_dim=None,
        bidirectional=False,
        batch_first=False,
    ):
        super(NavieComplexLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.input_dim = input_size // 2
        self.rnn_units = hidden_size // 2
        self.real_lstm = nn.LSTM(
            self.input_dim,
            self.rnn_units,
            num_layers=1,
            bidirectional=bidirectional,
            batch_first=False,
        )
        self.imag_lstm = nn.LSTM(
            self.input_dim,
            self.rnn_units,
            num_layers=1,
            bidirectional=bidirectional,
            batch_first=False,
        )
        if bidirectional:
            bidirectional = 2
        else:
            bidirectional = 1
        if projection_dim is not None:
            self.projection_dim = projection_dim // 2
            self.r_trans = nn.Linear(
                self.rnn_units * bidirectional, self.projection_dim
            )
            self.i_trans = nn.Linear(
                self.rnn_units * bidirectional, self.projection_dim
            )
        else:
            self.projection_dim = None

    def forward(self, inputs):
        """
        Computes the forward pass of the NavieComplexLSTM.

    This method takes complex-valued input and processes it through two 
    separate LSTM layers for the real and imaginary parts. It then combines 
    the outputs to produce the final complex output.

    Args:
        inputs (Union[torch.Tensor, List[torch.Tensor]]): A tensor or a list 
            containing the real and imaginary parts of the input. If a tensor, 
            it should be of shape (seq_len, batch_size, input_size) where 
            input_size is the total number of input features (real + imaginary). 
            If a list, it should contain two tensors: the first for the real 
            part and the second for the imaginary part.

    Returns:
        List[torch.Tensor]: A list containing two tensors: the processed real 
        and imaginary parts. Each tensor will have the shape 
        (seq_len, batch_size, output_size) where output_size is determined 
        by the hidden size and whether projection_dim is set.

    Examples:
        >>> lstm = NavieComplexLSTM(input_size=4, hidden_size=4)
        >>> real_input = torch.randn(10, 2, 2)  # (seq_len, batch_size, real_dim)
        >>> imag_input = torch.randn(10, 2, 2)  # (seq_len, batch_size, imag_dim)
        >>> outputs = lstm([real_input, imag_input])
        >>> real_out, imag_out = outputs
        >>> print(real_out.shape)  # Should match (10, 2, output_size)
        >>> print(imag_out.shape)  # Should match (10, 2, output_size)

    Note:
        The input size must be divisible by 2, as it expects the real and 
        imaginary parts to be interleaved.

    Raises:
        ValueError: If the input dimensions do not match the expected 
        input shape or if the input size is not divisible by 2.
        """
        if isinstance(inputs, list):
            real, imag = inputs
        elif isinstance(inputs, torch.Tensor):
            real, imag = torch.chunk(inputs, -1)
        r2r_out = self.real_lstm(real)[0]
        r2i_out = self.imag_lstm(real)[0]
        i2r_out = self.real_lstm(imag)[0]
        i2i_out = self.imag_lstm(imag)[0]
        real_out = r2r_out - i2i_out
        imag_out = i2r_out + r2i_out
        if self.projection_dim is not None:
            real_out = self.r_trans(real_out)
            imag_out = self.i_trans(imag_out)
        return [real_out, imag_out]

    def flatten_parameters(self):
        """
        Flatten the parameters of the LSTM layers for efficient training.

    This method is particularly useful for optimizing the performance of LSTM
    layers when using packed sequences, as it allows the LSTMs to use a single
    contiguous memory block for their weights.

    The method calls `flatten_parameters()` on both the real and imaginary
    LSTM layers to ensure their parameters are properly flattened.

    Attributes:
        real_lstm (nn.LSTM): The real-valued LSTM layer.
        imag_lstm (nn.LSTM): The imaginary-valued LSTM layer.

    Examples:
        >>> model = NavieComplexLSTM(input_size=4, hidden_size=8)
        >>> model.flatten_parameters()

    Note:
        This method should be called before training when using packed sequences
        to ensure optimal performance.

    Raises:
        RuntimeError: If the LSTM layers have not been properly initialized.
        """
        self.real_lstm.flatten_parameters()


def complex_cat(inputs, axis):
    """
    Concatenate complex-valued tensors along a specified axis.

    This function takes a list of complex-valued tensors, splits each tensor
    into its real and imaginary parts, concatenates these parts along the
    specified axis, and returns a single complex-valued tensor.

    Attributes:
        inputs (list): A list of complex-valued tensors, where each tensor is 
            expected to have an even number of channels (real and imaginary parts).
        axis (int): The axis along which to concatenate the real and imaginary 
            parts.

    Args:
        inputs (list of torch.Tensor): A list containing complex-valued tensors.
        axis (int): The axis along which to concatenate the real and imaginary 
            parts.

    Returns:
        torch.Tensor: A single complex-valued tensor formed by concatenating the 
            real and imaginary parts of the input tensors along the specified 
            axis.

    Examples:
        >>> import torch
        >>> a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # Real part
        >>> b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])  # Imaginary part
        >>> complex_tensor = torch.cat([a, b], dim=-1)  # Combine real and imag
        >>> result = complex_cat([complex_tensor, complex_tensor], axis=0)
        >>> print(result)
        tensor([[1., 2., 5., 6.],
                [3., 4., 7., 8.],
                [1., 2., 5., 6.],
                [3., 4., 7., 8.]])
    
    Note:
        The input tensors must have the same shape along all dimensions 
        except for the specified axis.
    """
    real, imag = [], []
    for idx, data in enumerate(inputs):
        r, i = torch.chunk(data, 2, axis)
        real.append(r)
        imag.append(i)
    real = torch.cat(real, axis)
    imag = torch.cat(imag, axis)
    outputs = torch.cat([real, imag], axis)
    return outputs


class ComplexConv2d(nn.Module):
    """
    Complex 2D convolution layer for processing complex-valued inputs.

    This layer performs convolution on complex inputs, where the input is
    represented as two separate channels: real and imaginary parts. It applies
    two separate 2D convolution operations, one for the real part and one for
    the imaginary part, and combines the results to produce the output.

    Attributes:
        in_channels (int): Number of input channels (real + imag).
        out_channels (int): Number of output channels (real + imag).
        kernel_size (tuple): Size of the convolution kernel.
        stride (tuple): Stride of the convolution.
        padding (tuple): Padding applied to the input.
        causal (bool): If True, applies causal padding on the time dimension.
        groups (int): Number of blocked connections from input channels to 
            output channels.
        dilation (int): Spacing between kernel elements.
        complex_axis (int): Axis along which the real and imaginary parts are 
            concatenated.

    Args:
        in_channels (int): Number of input channels (real + imag).
        out_channels (int): Number of output channels (real + imag).
        kernel_size (tuple, optional): Size of the convolution kernel (default: (1, 1)).
        stride (tuple, optional): Stride of the convolution (default: (1, 1)).
        padding (tuple, optional): Padding applied to the input (default: (0, 0)).
        dilation (int, optional): Spacing between kernel elements (default: 1).
        groups (int, optional): Number of blocked connections from input channels 
            to output channels (default: 1).
        causal (bool, optional): If True, applies causal padding on the time dimension 
            (default: True).
        complex_axis (int, optional): Axis along which the real and imaginary parts 
            are concatenated (default: 1).

    Returns:
        torch.Tensor: Output tensor containing concatenated real and imaginary 
        parts after convolution.

    Examples:
        >>> import torch
        >>> conv = ComplexConv2d(in_channels=4, out_channels=8, kernel_size=(3, 3))
        >>> input_tensor = torch.randn(1, 4, 10, 10)  # Shape: [B, C, D, T]
        >>> output = conv(input_tensor)
        >>> output.shape
        torch.Size([1, 8, 8, 8])  # Output shape after convolution

    Note:
        The input tensor is expected to have a shape of [B, C, D, T], where
        B is the batch size, C is the number of channels (real + imag),
        D is the height, and T is the width of the input.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=1,
        groups=1,
        causal=True,
        complex_axis=1,
    ):
        """ComplexConv2d.

        in_channels: real+imag
        out_channels: real+imag
        kernel_size : input [B,C,D,T] kernel size in [D,T]
        padding : input [B,C,D,T] padding in [D,T]
        causal: if causal, will padding time dimension's left side,
                otherwise both
        """
        super(ComplexConv2d, self).__init__()
        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.causal = causal
        self.groups = groups
        self.dilation = dilation
        self.complex_axis = complex_axis
        self.real_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size,
            self.stride,
            padding=[self.padding[0], 0],
            dilation=self.dilation,
            groups=self.groups,
        )
        self.imag_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size,
            self.stride,
            padding=[self.padding[0], 0],
            dilation=self.dilation,
            groups=self.groups,
        )

        nn.init.normal_(self.real_conv.weight.data, std=0.05)
        nn.init.normal_(self.imag_conv.weight.data, std=0.05)
        nn.init.constant_(self.real_conv.bias, 0.0)
        nn.init.constant_(self.imag_conv.bias, 0.0)

    def forward(self, inputs):
        """
        Forward pass for the ComplexConv2d layer.

    This method performs the forward computation of the complex convolution
    layer, applying separate convolutions for the real and imaginary parts of
    the input. The results from the real and imaginary convolutions are then
    combined to produce the final output.

    Args:
        inputs (torch.Tensor): Input tensor of shape [B, C, D, T], where
            C represents the complex channels (real and imaginary), B is the
            batch size, D is the depth, and T is the time dimension.

    Returns:
        torch.Tensor: Output tensor of the same shape as the input, after
        applying the complex convolution operation.

    Note:
        If `self.padding[1]` is not zero and `self.causal` is True, the input
        is padded on the left side of the time dimension. Otherwise, it is
        padded symmetrically.

    Examples:
        >>> conv = ComplexConv2d(in_channels=4, out_channels=8, kernel_size=(3, 3))
        >>> input_tensor = torch.randn(1, 4, 10, 10)  # Example input
        >>> output = conv(input_tensor)
        >>> output.shape
        torch.Size([1, 8, 10, 10])  # Output shape after convolution
        """
        if self.padding[1] != 0 and self.causal:
            inputs = F.pad(inputs, [self.padding[1], 0, 0, 0])
        else:
            inputs = F.pad(inputs, [self.padding[1], self.padding[1], 0, 0])

        if self.complex_axis == 0:
            real = self.real_conv(inputs)
            imag = self.imag_conv(inputs)
            real2real, imag2real = torch.chunk(real, 2, self.complex_axis)
            real2imag, imag2imag = torch.chunk(imag, 2, self.complex_axis)

        else:
            if isinstance(inputs, torch.Tensor):
                real, imag = torch.chunk(inputs, 2, self.complex_axis)

            real2real = self.real_conv(
                real,
            )
            imag2imag = self.imag_conv(
                imag,
            )

            real2imag = self.imag_conv(real)
            imag2real = self.real_conv(imag)

        real = real2real - imag2imag
        imag = real2imag + imag2real
        out = torch.cat([real, imag], self.complex_axis)

        return out


class ComplexConvTranspose2d(nn.Module):
    """
    ComplexConvTranspose2d.

        This module performs a 2D transposed convolution operation on complex 
        inputs, which are expected to be represented as two channels: 
        real and imaginary. The input channels and output channels are 
        expected to be double the size (real + imag).

        Attributes:
            in_channels (int): Number of input channels (real + imag).
            out_channels (int): Number of output channels (real + imag).
            kernel_size (tuple): Size of the convolution kernel.
            stride (tuple): Stride of the convolution.
            padding (tuple): Padding added to both sides of the input.
            output_padding (tuple): Additional size added to one side of the 
                                    output shape.
            causal (bool): If True, pads the left side of the time dimension 
                           for causal convolutions.
            complex_axis (int): The axis along which to split the complex 
                                input into real and imaginary parts.
            groups (int): Number of groups for grouped convolution.

        Args:
            in_channels (int): Number of input channels (real + imag).
            out_channels (int): Number of output channels (real + imag).
            kernel_size (tuple, optional): Size of the convolution kernel.
            stride (tuple, optional): Stride of the convolution.
            padding (tuple, optional): Padding added to both sides of the input.
            output_padding (tuple, optional): Additional size added to one side 
                                               of the output shape.
            causal (bool, optional): If True, pads the left side of the time 
                                      dimension for causal convolutions.
            complex_axis (int, optional): The axis along which to split the 
                                           complex input into real and 
                                           imaginary parts.
            groups (int, optional): Number of groups for grouped convolution.

        Returns:
            torch.Tensor: The output tensor containing concatenated real and 
                          imaginary parts after transposed convolution.

        Examples:
            >>> conv_transpose = ComplexConvTranspose2d(
            ...     in_channels=4, out_channels=8, kernel_size=(3, 3), stride=(2, 2)
            ... )
            >>> input_tensor = torch.randn(1, 8, 10, 10)  # Batch size 1, 8 channels
            >>> output = conv_transpose(input_tensor)
            >>> output.shape
            torch.Size([1, 16, 20, 20])  # Output shape after transposed conv

        Note:
            The input tensor must be structured such that the real and 
            imaginary parts are split along the complex axis.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        output_padding=(0, 0),
        causal=False,
        complex_axis=1,
        groups=1,
    ):
        """ComplexConvTranspose2d.

        in_channels: real+imag
        out_channels: real+imag
        """
        super(ComplexConvTranspose2d, self).__init__()
        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        self.real_conv = nn.ConvTranspose2d(
            self.in_channels,
            self.out_channels,
            kernel_size,
            self.stride,
            padding=self.padding,
            output_padding=output_padding,
            groups=self.groups,
        )
        self.imag_conv = nn.ConvTranspose2d(
            self.in_channels,
            self.out_channels,
            kernel_size,
            self.stride,
            padding=self.padding,
            output_padding=output_padding,
            groups=self.groups,
        )
        self.complex_axis = complex_axis

        nn.init.normal_(self.real_conv.weight, std=0.05)
        nn.init.normal_(self.imag_conv.weight, std=0.05)
        nn.init.constant_(self.real_conv.bias, 0.0)
        nn.init.constant_(self.imag_conv.bias, 0.0)

    def forward(self, inputs):
        """
        Applies the complex transposed convolution to the input tensor.

    This method takes an input tensor, splits it into real and imaginary
    parts, and then applies the complex transposed convolution operation
    to these parts separately. The results are then combined to form the
    output tensor.

    Args:
        inputs (torch.Tensor or tuple or list): The input tensor containing
            real and imaginary components, which can be in the form of a
            single tensor or a tuple/list of two tensors.

    Returns:
        torch.Tensor: A tensor that contains the output of the complex
        transposed convolution, with the same format as the input (real and
        imaginary parts concatenated).

    Examples:
        >>> import torch
        >>> conv_transpose = ComplexConvTranspose2d(4, 8, kernel_size=(3, 3))
        >>> input_tensor = torch.randn(1, 8, 10, 10)  # Batch size of 1
        >>> output = conv_transpose(input_tensor)
        >>> output.shape
        torch.Size([1, 16, 12, 12])  # Adjusted dimensions after transposed conv

    Note:
        The input tensor is expected to be of shape [B, C, H, W] where B is
        the batch size, C is the number of channels (real + imaginary), H is
        the height, and W is the width.

    Raises:
        ValueError: If the input tensor does not have the correct number of
        channels (must be even, as they represent real and imaginary parts).

    Todo:
        - Add support for different padding modes in the future.
        """
        if isinstance(inputs, torch.Tensor):
            real, imag = torch.chunk(inputs, 2, self.complex_axis)
        elif isinstance(inputs, tuple) or isinstance(inputs, list):
            real = inputs[0]
            imag = inputs[1]
        if self.complex_axis == 0:
            real = self.real_conv(inputs)
            imag = self.imag_conv(inputs)
            real2real, imag2real = torch.chunk(real, 2, self.complex_axis)
            real2imag, imag2imag = torch.chunk(imag, 2, self.complex_axis)

        else:
            if isinstance(inputs, torch.Tensor):
                real, imag = torch.chunk(inputs, 2, self.complex_axis)

            real2real = self.real_conv(
                real,
            )
            imag2imag = self.imag_conv(
                imag,
            )

            real2imag = self.imag_conv(real)
            imag2real = self.real_conv(imag)

        real = real2real - imag2imag
        imag = real2imag + imag2real

        out = torch.cat([real, imag], self.complex_axis)

        return out


class ComplexBatchNorm(torch.nn.Module):
    """
    Applies Batch Normalization over a complex input.

    This layer normalizes the input based on the mean and variance of 
    the batch. It operates on complex-valued data by separately 
    normalizing the real and imaginary parts. The layer supports 
    learnable affine parameters.

    Attributes:
        num_features (int): Number of features in the input (real + imag).
        eps (float): A small constant added to the denominator for 
            numerical stability.
        momentum (float): Momentum for the running mean and variance.
        affine (bool): If True, this layer has learnable parameters.
        track_running_stats (bool): If True, this layer tracks 
            running statistics.
        complex_axis (int): The axis along which the complex parts 
            are separated.

    Args:
        num_features (int): Number of features in the input (real + imag).
        eps (float, optional): A small constant added to the denominator 
            for numerical stability. Defaults to 1e-5.
        momentum (float, optional): Momentum for the running mean and 
            variance. Defaults to 0.1.
        affine (bool, optional): If True, this layer has learnable 
            parameters. Defaults to True.
        track_running_stats (bool, optional): If True, this layer 
            tracks running statistics. Defaults to True.
        complex_axis (int, optional): The axis along which the complex 
            parts are separated. Defaults to 1.

    Examples:
        >>> import torch
        >>> layer = ComplexBatchNorm(num_features=4)
        >>> input_tensor = torch.randn(2, 4, 3)  # Batch of 2, 4 features, 3 time steps
        >>> output = layer(input_tensor)

    Note:
        The layer normalizes the real and imaginary parts separately and 
        supports affine transformation for learnable parameters.

    Raises:
        AssertionError: If the input dimensions are not compatible with 
            the expected dimensions.
    """
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        complex_axis=1,
    ):
        super(ComplexBatchNorm, self).__init__()
        self.num_features = num_features // 2
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.complex_axis = complex_axis

        if self.affine:
            self.Wrr = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Wri = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Wii = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Br = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Bi = torch.nn.Parameter(torch.Tensor(self.num_features))
        else:
            self.register_parameter("Wrr", None)
            self.register_parameter("Wri", None)
            self.register_parameter("Wii", None)
            self.register_parameter("Br", None)
            self.register_parameter("Bi", None)

        if self.track_running_stats:
            self.register_buffer("RMr", torch.zeros(self.num_features))
            self.register_buffer("RMi", torch.zeros(self.num_features))
            self.register_buffer("RVrr", torch.ones(self.num_features))
            self.register_buffer("RVri", torch.zeros(self.num_features))
            self.register_buffer("RVii", torch.ones(self.num_features))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("RMr", None)
            self.register_parameter("RMi", None)
            self.register_parameter("RVrr", None)
            self.register_parameter("RVri", None)
            self.register_parameter("RVii", None)
            self.register_parameter("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self):
        """
        Reset the running statistics of the batch normalization.

    This method sets the running mean and running variance to their initial
    values, which are zeros for the running mean and ones for the running 
    variance. It also resets the count of batches tracked.

    Attributes:
        RMr (torch.Tensor): Running mean for the real part.
        RMi (torch.Tensor): Running mean for the imaginary part.
        RVrr (torch.Tensor): Running variance for the real-real part.
        RVri (torch.Tensor): Running variance for the real-imaginary part.
        RVii (torch.Tensor): Running variance for the imaginary-imaginary part.
        num_batches_tracked (torch.Tensor): Count of batches processed.

    Note:
        This method is typically called when you want to reinitialize the
        statistics, for instance, at the start of a new training phase.

    Examples:
        >>> batch_norm = ComplexBatchNorm(num_features=4)
        >>> batch_norm.reset_running_stats()
        >>> print(batch_norm.RMr)  # Should print a tensor of zeros
        >>> print(batch_norm.RVrr)  # Should print a tensor of ones
        """
        if self.track_running_stats:
            self.RMr.zero_()
            self.RMi.zero_()
            self.RVrr.fill_(1)
            self.RVri.zero_()
            self.RVii.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        """
        Resets the parameters of the ComplexBatchNorm layer.

    This method resets the running statistics (mean and variance) to their
    initial values and initializes the learnable parameters (weights and
    biases) if affine transformation is enabled. The weights are initialized
    to ensure that the transformation is valid (e.g., positive-definite for
    covariance).

    Attributes:
        Br (torch.nn.Parameter): Bias for the real part.
        Bi (torch.nn.Parameter): Bias for the imaginary part.
        Wrr (torch.nn.Parameter): Weight for the real-real part.
        Wri (torch.nn.Parameter): Weight for the real-imaginary part.
        Wii (torch.nn.Parameter): Weight for the imaginary-imaginary part.

    Note:
        This method should be called after the initialization of the layer to
        ensure all parameters are set to their starting values.

    Examples:
        >>> batch_norm = ComplexBatchNorm(num_features=4)
        >>> batch_norm.reset_parameters()
        >>> print(batch_norm.Wrr)
        tensor([1., 1., 1., 1.])
        >>> print(batch_norm.Br)
        tensor([0., 0., 0., 0.])
        """
        if self.affine:
            self.Br.data.zero_()
            self.Bi.data.zero_()
            self.Wrr.data.fill_(1)
            self.Wri.data.uniform_(-0.9, +0.9)  # W will be positive-definite
            self.Wii.data.fill_(1)

    def _check_input_dim(self, xr, xi):
        assert xr.shape == xi.shape
        assert xr.size(1) == self.num_features

    def forward(self, inputs):
        """
        Applies the complex batch normalization to the input tensor.

    This method performs batch normalization on the input tensor, which is 
    expected to be in a complex format, where the real and imaginary parts 
    are concatenated along a specified axis. The method computes the mean 
    and variance for both the real and imaginary parts and normalizes the 
    inputs accordingly. If the layer is in training mode and tracking 
    running statistics, it updates the running mean and variance.

    Args:
        inputs (torch.Tensor): A tensor containing the input data with 
            shape (..., 2 * num_features) where the last dimension 
            contains the real and imaginary parts.

    Returns:
        torch.Tensor: The normalized output tensor, which has the same 
            shape as the input tensor.

    Raises:
        AssertionError: If the dimensions of the real and imaginary parts 
            do not match or if the input tensor does not have the correct 
            number of features.

    Examples:
        >>> batch_norm = ComplexBatchNorm(num_features=4)
        >>> input_tensor = torch.randn(10, 8)  # 10 samples, 4 real + 4 imag
        >>> output = batch_norm(input_tensor)

    Note:
        This method uses the `momentum` parameter for updating running 
        statistics if the layer is in training mode. If `momentum` is 
        set to None, a cumulative moving average is used instead.
        """
        xr, xi = torch.chunk(inputs, 2, axis=self.complex_axis)
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        # NOTE: The precise meaning of the "training flag" is:
        #       True:  Normalize using batch   statistics, update running statistics
        #              if they are being collected.
        #       False: Normalize using running statistics, ignore batch   statistics.
        training = self.training or not self.track_running_stats
        redux = [i for i in reversed(range(xr.dim())) if i != 1]
        vdim = [1] * xr.dim()
        vdim[1] = xr.size(1)

        # Mean M Computation and Centering
        # Includes running mean update if training and running.
        if training:
            Mr, Mi = xr, xi
            for d in redux:
                Mr = Mr.mean(d, keepdim=True)
                Mi = Mi.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RMr.lerp_(Mr.squeeze(), exponential_average_factor)
                self.RMi.lerp_(Mi.squeeze(), exponential_average_factor)
        else:
            Mr = self.RMr.view(vdim)
            Mi = self.RMi.view(vdim)
        xr, xi = xr - Mr, xi - Mi

        # Variance Matrix V Computation
        # Includes epsilon numerical stabilizer/Tikhonov regularizer.
        # Includes running variance update if training and running.
        if training:
            Vrr = xr * xr
            Vri = xr * xi
            Vii = xi * xi
            for d in redux:
                Vrr = Vrr.mean(d, keepdim=True)
                Vri = Vri.mean(d, keepdim=True)
                Vii = Vii.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RVrr.lerp_(Vrr.squeeze(), exponential_average_factor)
                self.RVri.lerp_(Vri.squeeze(), exponential_average_factor)
                self.RVii.lerp_(Vii.squeeze(), exponential_average_factor)
        else:
            Vrr = self.RVrr.view(vdim)
            Vri = self.RVri.view(vdim)
            Vii = self.RVii.view(vdim)
        Vrr = Vrr + self.eps
        Vri = Vri
        Vii = Vii + self.eps

        # Matrix Inverse Square Root U = V^-0.5
        # sqrt of a 2x2 matrix,
        # - https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
        tau = Vrr + Vii
        delta = torch.addcmul(Vrr * Vii, -1, Vri, Vri)
        s = delta.sqrt()
        t = (tau + 2 * s).sqrt()

        # matrix inverse, http://mathworld.wolfram.com/MatrixInverse.html
        rst = (s * t).reciprocal()
        Urr = (s + Vii) * rst
        Uii = (s + Vrr) * rst
        Uri = (-Vri) * rst

        # Optionally left-multiply U by affine weights W to produce combined
        # weights Z, left-multiply the inputs by Z, then optionally bias them.
        #
        # y = Zx + B
        # y = WUx + B
        # y = [Wrr Wri][Urr Uri] [xr] + [Br]
        #     [Wir Wii][Uir Uii] [xi]   [Bi]
        if self.affine:
            Wrr, Wri, Wii = (
                self.Wrr.view(vdim),
                self.Wri.view(vdim),
                self.Wii.view(vdim),
            )
            Zrr = (Wrr * Urr) + (Wri * Uri)
            Zri = (Wrr * Uri) + (Wri * Uii)
            Zir = (Wri * Urr) + (Wii * Uri)
            Zii = (Wri * Uri) + (Wii * Uii)
        else:
            Zrr, Zri, Zir, Zii = Urr, Uri, Uri, Uii

        yr = (Zrr * xr) + (Zri * xi)
        yi = (Zir * xr) + (Zii * xi)

        if self.affine:
            yr = yr + self.Br.view(vdim)
            yi = yi + self.Bi.view(vdim)

        outputs = torch.cat([yr, yi], self.complex_axis)
        return outputs

    def extra_repr(self):
        """
        Class for applying complex batch normalization to complex-valued inputs.

    This class implements complex batch normalization, which normalizes both the
    real and imaginary parts of the input. It maintains running statistics for
    training and evaluation modes.

    Attributes:
        num_features (int): The number of features in the input (real + imag).
        eps (float): A value added to the denominator for numerical stability.
        momentum (float): The value used for the running_mean and running_var
            computation.
        affine (bool): If True, this module has learnable parameters.
        track_running_stats (bool): If True, tracks the running mean and variance.
        complex_axis (int): The axis along which the complex parts are split.

    Args:
        num_features (int): Number of features (real + imag) in the input.
        eps (float, optional): Default is 1e-5.
        momentum (float, optional): Default is 0.1.
        affine (bool, optional): Default is True.
        track_running_stats (bool, optional): Default is True.
        complex_axis (int, optional): Default is 1.

    Methods:
        reset_running_stats(): Resets the running mean and variance.
        reset_parameters(): Resets learnable parameters and running stats.
        forward(inputs): Applies complex batch normalization to the input.

    Examples:
        >>> batch_norm = ComplexBatchNorm(num_features=4)
        >>> input_tensor = torch.randn(2, 4, 10)  # (batch_size, features, time)
        >>> output_tensor = batch_norm(input_tensor)

    Note:
        The `forward` method will compute batch normalization for the real and
        imaginary parts of the input tensor separately.

    Todo:
        - Add support for additional features if required.
        """
        return (
            "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**self.__dict__)
        )
