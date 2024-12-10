# Implementation of the TCN proposed in
# Luo. et al.  "Conv-tasnet: Surpassing ideal time–frequency
# magnitude masking for speech separation."
#
# The code is based on:
# https://github.com/kaituoxu/Conv-TasNet/blob/master/src/conv_tasnet.py
#


import torch
import torch.nn as nn

EPS = torch.finfo(torch.get_default_dtype()).eps


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network for speech separation.

    This class implements the Temporal Convolutional Network (TCN) as proposed in
    Luo et al. in the paper "Conv-tasnet: Surpassing ideal time–frequency magnitude
    masking for speech separation". The TCN is designed to process time-series data
    and is utilized in the context of speech separation tasks.

    Attributes:
        network: A sequential container comprising layer normalization,
                 a bottleneck convolution, and multiple temporal blocks.

    Args:
        N (int): Number of filters in the autoencoder.
        B (int): Number of channels in the bottleneck 1x1 convolution block.
        H (int): Number of channels in convolutional blocks.
        P (int): Kernel size in convolutional blocks.
        X (int): Number of convolutional blocks in each repeat.
        R (int): Number of repeats of the block structure.
        norm_type (str): Normalization type, can be 'BN', 'gLN', or 'cLN'.
        causal (bool): If True, applies causal convolutions; otherwise, applies
                       non-causal convolutions.

    Returns:
        bottleneck_feature: A tensor of shape [M, B, K] where M is the batch size,
                           B is the number of bottleneck channels, and K is the
                           length of the input sequences.

    Examples:
        >>> model = TemporalConvNet(N=64, B=16, H=32, P=3, X=4, R=2)
        >>> mixture_w = torch.randn(10, 64, 100)  # Batch of 10, 64 channels, length 100
        >>> output = model(mixture_w)
        >>> output.shape
        torch.Size([10, 16, 100])  # Output shape after processing

    Note:
        The output length will remain consistent with the input length if
        appropriate padding is used.

    Raises:
        ValueError: If `norm_type` is not one of 'BN', 'gLN', or 'cLN'.
    """

    def __init__(self, N, B, H, P, X, R, norm_type="gLN", causal=False):
        """Basic Module of tasnet.

        Args:
            N: Number of filters in autoencoder
            B: Number of channels in bottleneck 1 * 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
        """
        super().__init__()
        # Components
        # [M, N, K] -> [M, N, K]
        layer_norm = ChannelwiseLayerNorm(N)
        # [M, N, K] -> [M, B, K]
        bottleneck_conv1x1 = nn.Conv1d(N, B, 1, bias=False)
        # [M, B, K] -> [M, B, K]
        repeats = []
        for r in range(R):
            blocks = []
            for x in range(X):
                dilation = 2**x
                padding = (P - 1) * dilation if causal else (P - 1) * dilation // 2
                blocks += [
                    TemporalBlock(
                        B,
                        H,
                        P,
                        stride=1,
                        padding=padding,
                        dilation=dilation,
                        norm_type=norm_type,
                        causal=causal,
                    )
                ]
            repeats += [nn.Sequential(*blocks)]
        temporal_conv_net = nn.Sequential(*repeats)
        # Put together (except mask_conv1x1, modified from the original code)
        self.network = nn.Sequential(layer_norm, bottleneck_conv1x1, temporal_conv_net)

    def forward(self, mixture_w):
        """
        Forward pass for the TemporalConvNet.

        This method processes the input mixture of waveforms through the
        temporal convolutional network to produce bottleneck features.

        Args:
            mixture_w (torch.Tensor): Input tensor of shape [M, N, K],
            where M is the batch size, N is the number of filters,
            and K is the sequence length.

        Returns:
            torch.Tensor: Output tensor of shape [M, B, K], where B is
            the number of channels in the bottleneck layer.

        Examples:
            >>> model = TemporalConvNet(N=64, B=32, H=128, P=3, X=4, R=2)
            >>> mixture = torch.randn(16, 64, 100)  # Batch of 16, 64 filters, 100 length
            >>> output = model(mixture)
            >>> print(output.shape)
            torch.Size([16, 32, 100])  # Expected output shape

        Note:
            Ensure that the input tensor is properly shaped as specified
            to avoid dimension errors during processing.
        """
        return self.network(mixture_w)  # [M, N, K] -> [M, B, K]


class TemporalBlock(nn.Module):
    """
    Temporal Block for use in Temporal Convolutional Networks (TCN).

    This class implements a single block of a Temporal Convolutional Network,
    which consists of a 1x1 convolution, a PReLU activation, a normalization
    layer, and a depthwise separable convolution. The block also includes a
    residual connection that adds the input to the output of the block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride for the convolution.
        padding (int): Padding for the convolution.
        dilation (int): Dilation factor for the convolution.
        norm_type (str, optional): Type of normalization to use. Can be
            "gLN" (global layer normalization), "cLN" (channel-wise layer
            normalization), or "BN" (batch normalization). Default is "gLN".
        causal (bool, optional): If True, the convolution is causal. Default is False.

    Returns:
        Tensor: Output tensor of shape [M, B, K], where M is batch size,
        B is number of output channels, and K is sequence length.

    Examples:
        >>> block = TemporalBlock(in_channels=16, out_channels=32, kernel_size=3,
        ...                        stride=1, padding=1, dilation=1)
        >>> input_tensor = torch.randn(10, 16, 50)  # [M, C, K]
        >>> output_tensor = block(input_tensor)
        >>> output_tensor.shape
        torch.Size([10, 32, 50])  # Output shape

    Note:
        The residual connection adds the input tensor to the output of the
        network. This may help in training deeper networks by alleviating
        the vanishing gradient problem.

    Todo:
        - Investigate padding requirements for different kernel sizes.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        norm_type="gLN",
        causal=False,
    ):
        super().__init__()
        # [M, B, K] -> [M, H, K]
        conv1x1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        prelu = nn.PReLU()
        norm = chose_norm(norm_type, out_channels)
        # [M, H, K] -> [M, B, K]
        dsconv = DepthwiseSeparableConv(
            out_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            norm_type,
            causal,
        )
        # Put together
        self.net = nn.Sequential(conv1x1, prelu, norm, dsconv)

    def forward(self, x):
        """Forward.

        Args:
            x: [M, B, K]

        Returns:
            [M, B, K]
        """
        residual = x
        out = self.net(x)
        # TODO(Jing): when P = 3 here works fine, but when P = 2 maybe need to pad?
        return out + residual  # look like w/o F.relu is better than w/ F.relu
        # return F.relu(out + residual)


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution Layer.

    This class implements a depthwise separable convolution, which consists
    of a depthwise convolution followed by a pointwise convolution. The
    depthwise convolution applies a single filter per input channel, while
    the pointwise convolution mixes the outputs of the depthwise layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride of the convolution.
        padding (int): Padding added to both sides of the input.
        dilation (int): Dilation factor for the convolution.
        norm_type (str): Type of normalization to use ('gLN', 'cLN', 'BN').
        causal (bool): If True, applies causal convolution (output at time t
                       does not depend on future time steps).

    Attributes:
        net (nn.Sequential): Sequential container of depthwise and pointwise
                            convolutions along with normalization and activation.

    Returns:
        result (Tensor): Output tensor after applying the depthwise separable
                         convolution.

    Examples:
        >>> model = DepthwiseSeparableConv(in_channels=64, out_channels=128,
        ...                                  kernel_size=3, stride=1, padding=1,
        ...                                  dilation=1, norm_type='gLN', causal=False)
        >>> input_tensor = torch.randn(32, 64, 100)  # Batch of 32, 64 channels, 100 length
        >>> output_tensor = model(input_tensor)
        >>> output_tensor.shape
        torch.Size([32, 128, 100])  # Output will have 128 channels

    Note:
        The depthwise convolution is performed with `groups` set to
        `in_channels` to achieve depthwise separability. If `causal` is
        set to True, a Chomp layer is used to ensure the output length
        matches the input length.

    Raises:
        ValueError: If an unsupported normalization type is specified.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        norm_type="gLN",
        causal=False,
    ):
        super().__init__()
        # Use `groups` option to implement depthwise convolution
        # [M, H, K] -> [M, H, K]
        depthwise_conv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        if causal:
            chomp = Chomp1d(padding)
        prelu = nn.PReLU()
        norm = chose_norm(norm_type, in_channels)
        # [M, H, K] -> [M, B, K]
        pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        # Put together
        if causal:
            self.net = nn.Sequential(depthwise_conv, chomp, prelu, norm, pointwise_conv)
        else:
            self.net = nn.Sequential(depthwise_conv, prelu, norm, pointwise_conv)

    def forward(self, x):
        """
        Forward pass for the TemporalConvNet.

        This method takes the input tensor `mixture_w` and passes it through the
        temporal convolution network to extract bottleneck features. The expected
        input shape is [M, N, K], where M is the batch size, N is the number of
        filters in the autoencoder, and K is the sequence length. The output will
        be of shape [M, B, K], where B is the number of channels in the bottleneck
        1x1-conv block.

        Args:
            mixture_w: A tensor of shape [M, N, K], representing the input
                       mixture signal, where M is the batch size, N is the
                       number of channels, and K is the length of the signal.

        Returns:
            bottleneck_feature: A tensor of shape [M, B, K], representing the
                                extracted bottleneck features after passing
                                through the network.

        Examples:
            >>> model = TemporalConvNet(N=64, B=32, H=128, P=3, X=4, R=3)
            >>> mixture = torch.randn(10, 64, 100)  # Batch of 10, 64 channels, length 100
            >>> output = model(mixture)
            >>> print(output.shape)
            torch.Size([10, 32, 100])  # Output shape should match [M, B, K]

        Note:
            Ensure that the input tensor `mixture_w` is correctly shaped
            according to the specifications to avoid dimension mismatch errors.
        """
        return self.net(x)


class Chomp1d(nn.Module):
    """
    To ensure the output length is the same as the input.

    This module removes a specified number of elements from the end of the input
    tensor, which is useful in convolutional architectures to maintain the
    desired output size after applying causal convolutions.

    Attributes:
        chomp_size (int): The number of elements to remove from the end of the input.

    Args:
        chomp_size (int): The size of the chomp, i.e., the number of elements to
            discard from the end of the input tensor.

    Returns:
        torch.Tensor: The output tensor with the last `chomp_size` elements removed.

    Examples:
        >>> chomp = Chomp1d(chomp_size=2)
        >>> input_tensor = torch.randn(5, 10, 20)  # [M, H, Kpad]
        >>> output_tensor = chomp(input_tensor)
        >>> output_tensor.shape
        torch.Size([5, 10, 18])  # Output shape after chomp

    Note:
        This module is particularly useful when used in conjunction with
        depthwise separable convolutions to maintain the appropriate sequence
        length for subsequent layers.
    """

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        To ensure the output length is the same as the input.

        This module removes a specified number of elements from the end of the
        input tensor along the last dimension, ensuring that the output has the
        same length as the original input minus the `chomp_size`.

        Attributes:
            chomp_size (int): The number of elements to remove from the end of the
                input tensor.

        Args:
            chomp_size: The number of elements to be removed from the end of the
                input tensor.

        Returns:
            A tensor of shape [M, H, K] where K is the original length minus
            `chomp_size`.

        Examples:
            >>> chomp = Chomp1d(chomp_size=2)
            >>> input_tensor = torch.randn(10, 5, 20)  # [M, H, Kpad]
            >>> output_tensor = chomp(input_tensor)
            >>> output_tensor.shape
            torch.Size([10, 5, 18])  # [M, H, K]

        Note:
            This module is typically used in conjunction with causal convolutions
            to ensure that the output length matches the expected dimensions for
            further processing in a network.

        Raises:
            ValueError: If `chomp_size` is negative or greater than the input
                length.
        """
        return x[:, :, : -self.chomp_size].contiguous()


def check_nonlinear(nolinear_type):
    """
    Check if the specified nonlinear type is supported.

    This function verifies that the provided nonlinear type is one of the
    accepted types. If the type is not supported, it raises a ValueError.

    Args:
        nonlinear_type (str): The type of nonlinear function to check.
            Accepted values are "softmax" and "relu".

    Raises:
        ValueError: If the nonlinear_type is not one of the supported types.

    Examples:
        >>> check_nonlinear("softmax")  # No exception raised
        >>> check_nonlinear("relu")      # No exception raised
        >>> check_nonlinear("sigmoid")   # Raises ValueError: Unsupported nonlinear type
    """
    if nolinear_type not in ["softmax", "relu"]:
        raise ValueError("Unsupported nonlinear type")


def chose_norm(norm_type, channel_size):
    """
    Choose the appropriate normalization layer based on the given type.

    The input of normalization will be (M, C, K), where M is the batch size,
    C is the channel size, and K is the sequence length. The function supports
    different types of normalization: global layer normalization (gLN),
    channel-wise layer normalization (cLN), and batch normalization (BN).

    Args:
        norm_type (str): The type of normalization to use. Should be one of:
            "gLN" for Global Layer Normalization,
            "cLN" for Channel-wise Layer Normalization,
            "BN" for Batch Normalization.
        channel_size (int): The number of channels for normalization.

    Returns:
        nn.Module: The corresponding normalization layer.

    Raises:
        ValueError: If the `norm_type` is not one of the supported types.

    Examples:
        >>> norm_layer = chose_norm("gLN", 64)
        >>> output = norm_layer(torch.randn(10, 64, 100))

        >>> norm_layer = chose_norm("cLN", 32)
        >>> output = norm_layer(torch.randn(10, 32, 100))

        >>> norm_layer = chose_norm("BN", 128)
        >>> output = norm_layer(torch.randn(10, 128, 100))

    Note:
        This function is designed to provide a flexible way to select
        normalization methods in the context of temporal convolutional networks.
    """
    if norm_type == "gLN":
        return GlobalLayerNorm(channel_size)
    elif norm_type == "cLN":
        return ChannelwiseLayerNorm(channel_size)
    elif norm_type == "BN":
        # Given input (M, C, K), nn.BatchNorm1d(C) will accumulate statics
        # along M and K, so this BN usage is right.
        return nn.BatchNorm1d(channel_size)
    else:
        raise ValueError("Unsupported normalization type")


class ChannelwiseLayerNorm(nn.Module):
    """
    Channel-wise Layer Normalization (cLN).

    This layer normalizes the input across the channel dimension.
    It uses learnable parameters gamma and beta to scale and shift the
    normalized output. This is particularly useful in various
    neural network architectures to stabilize the learning process.

    Attributes:
        gamma: A learnable parameter for scaling, initialized to 1.
        beta: A learnable parameter for shifting, initialized to 0.

    Args:
        channel_size (int): The number of channels in the input tensor.

    Methods:
        reset_parameters: Resets the parameters gamma and beta to their
            initial values.
        forward: Applies channel-wise normalization to the input tensor.

    Examples:
        >>> layer_norm = ChannelwiseLayerNorm(channel_size=64)
        >>> input_tensor = torch.randn(32, 64, 100)  # [M, N, K]
        >>> output_tensor = layer_norm(input_tensor)
        >>> print(output_tensor.shape)  # Output: torch.Size([32, 64, 100])

    Note:
        The input tensor must have three dimensions: (batch_size,
        channel_size, sequence_length).
    """

    def __init__(self, channel_size):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        """
            Channel-wise Layer Normalization (cLN).

        This class implements channel-wise layer normalization, which normalizes
        the input tensor along the channel dimension. The normalization is
        achieved by computing the mean and variance for each channel across the
        batch and sequence dimensions. The normalized output is scaled and
        shifted using learnable parameters gamma and beta.

        Attributes:
            gamma (nn.Parameter): Learnable scaling parameter of shape (1, N, 1).
            beta (nn.Parameter): Learnable shifting parameter of shape (1, N, 1).

        Args:
            channel_size (int): The number of channels (N) in the input tensor.

        Examples:
            >>> layer_norm = ChannelwiseLayerNorm(channel_size=64)
            >>> input_tensor = torch.randn(32, 64, 100)  # [M, N, K]
            >>> output_tensor = layer_norm(input_tensor)  # Output shape: [32, 64, 100]

        Note:
            The normalization is applied to the input tensor using the formula:
            cLN_y = gamma * (y - mean) / sqrt(var + EPS) + beta, where mean and
            var are computed for each channel.

        Todo:
            Add more initialization options for gamma and beta if necessary.
        """
        self.beta.data.zero_()

    def forward(self, y):
        """
        Forward pass for the TemporalConvNet.

        This method takes a mixture of audio signals as input and passes it through
        the temporal convolutional network, returning the bottleneck features.

        Args:
            mixture_w: A tensor of shape [M, N, K], where:
                M is the batch size,
                N is the number of input channels,
                K is the sequence length.

        Returns:
            bottleneck_feature: A tensor of shape [M, B, K], where:
                B is the number of bottleneck channels.

        Examples:
            >>> model = TemporalConvNet(N=64, B=16, H=32, P=3, X=4, R=2)
            >>> mixture = torch.randn(8, 64, 100)  # Batch of 8 samples
            >>> output = model(mixture)
            >>> print(output.shape)
            torch.Size([8, 16, 100])
        """
        mean = torch.mean(y, dim=1, keepdim=True)  # [M, 1, K]
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)  # [M, 1, K]
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return cLN_y


class GlobalLayerNorm(nn.Module):
    """
    Global Layer Normalization (gLN).

    This module applies global layer normalization to the input tensor.
    It normalizes the input across all channels and spatial dimensions,
    ensuring that the output has a mean of zero and a variance of one.

    Attributes:
        gamma (nn.Parameter): Scale parameter for normalization, initialized to 1.
        beta (nn.Parameter): Shift parameter for normalization, initialized to 0.

    Args:
        channel_size (int): The number of channels in the input tensor.

    Returns:
        gLN_y (torch.Tensor): Normalized output tensor of shape [M, N, K],
        where M is the batch size, N is the channel size, and K is the length.

    Examples:
        >>> gLN = GlobalLayerNorm(channel_size=10)
        >>> input_tensor = torch.randn(32, 10, 50)  # Batch of 32, 10 channels, length 50
        >>> output_tensor = gLN(input_tensor)
        >>> output_tensor.shape
        torch.Size([32, 10, 50])

    Note:
        The normalization is performed as follows:
        gLN_y = gamma * (y - mean) / sqrt(var + EPS) + beta
        where mean and var are calculated over all channels and spatial dimensions.
    """

    def __init__(self, channel_size):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        """
        Global Layer Normalization (gLN).

        This class implements the global layer normalization technique, which normalizes
        the input tensor across both the channel and spatial dimensions. This helps to
        stabilize the training of deep learning models by reducing internal covariate
        shift.

        Attributes:
            gamma (nn.Parameter): Learnable scale parameter of shape [1, N, 1].
            beta (nn.Parameter): Learnable shift parameter of shape [1, N, 1].

        Methods:
            reset_parameters: Resets the parameters gamma and beta to their initial values.
            forward: Applies global layer normalization to the input tensor.

        Args:
            channel_size (int): The number of channels in the input tensor.

        Examples:
            >>> gLN = GlobalLayerNorm(channel_size=64)
            >>> input_tensor = torch.randn(32, 64, 128)  # [M, N, K]
            >>> output_tensor = gLN(input_tensor)
            >>> output_tensor.shape
            torch.Size([32, 64, 128])  # Output shape is the same as input shape

        Note:
            The input tensor should have the shape [M, N, K], where M is the batch size,
            N is the number of channels, and K is the sequence length.
        """
        self.beta.data.zero_()

    def forward(self, y):
        """
        Global Layer Normalization (gLN).

        This class implements global layer normalization for input tensors.
        It normalizes the input across both the channel and sequence dimensions,
        making it suitable for various deep learning tasks, especially in
        sequence modeling.

        Attributes:
            gamma (torch.Parameter): Learnable scale parameter with shape [1, N, 1].
            beta (torch.Parameter): Learnable shift parameter with shape [1, N, 1].

        Methods:
            reset_parameters(): Initializes the parameters of gamma and beta.
            forward(y): Applies global layer normalization to the input tensor.

        Args:
            channel_size (int): The size of the channel dimension of the input tensor.

        Examples:
            >>> gLN = GlobalLayerNorm(channel_size=64)
            >>> input_tensor = torch.randn(32, 64, 128)  # [M, N, K]
            >>> output_tensor = gLN(input_tensor)  # [M, N, K]

        Note:
            The input tensor `y` is expected to have the shape [M, N, K],
            where M is the batch size, N is the channel size, and K is the length.

        Raises:
            ValueError: If the input tensor does not have the expected shape.
        """
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)  # [M, 1, 1]
        var = (
            (torch.pow(y - mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        )
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y
