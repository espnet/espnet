# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common layers for defining score networks."""

import string
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.enh.layers.ncsnpp_utils.normalization import ConditionalInstanceNorm2dPlus


def get_act(config):
    """
        Get activation functions based on the specified configuration.

    This function returns a corresponding activation function from the PyTorch
    library based on the input string configuration. Supported activation
    functions include ELU, ReLU, Leaky ReLU, and Swish. If an unsupported
    configuration is provided, a NotImplementedError is raised.

    Args:
        config (str): The name of the activation function to retrieve.
                       Supported values are "elu", "relu", "lrelu", and "swish".

    Returns:
        nn.Module: The corresponding activation function as a PyTorch module.

    Raises:
        NotImplementedError: If the specified activation function is not supported.

    Examples:
        >>> act_fn = get_act("relu")
        >>> print(act_fn)
        ReLU()

        >>> act_fn = get_act("swish")
        >>> print(act_fn)
        SiLU()

        >>> act_fn = get_act("unknown")
        Traceback (most recent call last):
            ...
        NotImplementedError: activation function does not exist!
    """

    if config == "elu":
        return nn.ELU()
    elif config == "relu":
        return nn.ReLU()
    elif config == "lrelu":
        return nn.LeakyReLU(negative_slope=0.2)
    elif config == "swish":
        return nn.SiLU()
    else:
        raise NotImplementedError("activation function does not exist!")


def ncsn_conv1x1(
    in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1.0, padding=0
):
    """
    1x1 convolution. Same as NCSNv1/v2.

    This function creates a 1x1 convolutional layer with specified parameters
    and initializes its weights and biases. The initialization is scaled by
    the `init_scale` parameter. If `init_scale` is zero, it defaults to a
    very small value to avoid zero initialization.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int, optional): Stride of the convolution. Default is 1.
        bias (bool, optional): If True, adds a learnable bias to the output.
            Default is True.
        dilation (int, optional): Spacing between kernel elements. Default is 1.
        init_scale (float, optional): Scale factor for weight initialization.
            Default is 1.0.
        padding (int, optional): Implicit zero padding to be added on both
            sides. Default is 0.

    Returns:
        torch.nn.Conv2d: A 1x1 convolutional layer with initialized weights
        and biases.

    Examples:
        >>> conv_layer = ncsn_conv1x1(in_planes=64, out_planes=128, stride=1)
        >>> print(conv_layer)
        Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))

    Note:
        The weight and bias are initialized to `init_scale`. If `init_scale`
        is set to zero, it is overridden to a very small value (1e-10) to
        prevent zero initialization.

    Raises:
        ValueError: If any argument is invalid (not applicable in this case).
    """
    conv = nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=bias,
        dilation=dilation,
        padding=padding,
    )
    init_scale = 1e-10 if init_scale == 0 else init_scale
    conv.weight.data *= init_scale
    conv.bias.data *= init_scale
    return conv


def variance_scaling(
    scale, mode, distribution, in_axis=1, out_axis=0, dtype=torch.float32, device="cpu"
):
    """
    Initialize a tensor using variance scaling.

    This function provides a variance scaling initializer, which can be useful
    for initializing weights in neural networks. It computes the fan-in and
    fan-out of the tensor shape and scales the variance according to the specified
    mode and distribution.

    Args:
        scale (float): Scaling factor for the variance.
        mode (str): One of {'fan_in', 'fan_out', 'fan_avg'} that determines how
            to calculate the variance.
        distribution (str): One of {'normal', 'uniform'} that determines the
            distribution of the initialized values.
        in_axis (int, optional): Axis that corresponds to input dimension.
            Defaults to 1.
        out_axis (int, optional): Axis that corresponds to output dimension.
            Defaults to 0.
        dtype (torch.dtype, optional): Desired data type of the tensor. Defaults
            to torch.float32.
        device (str, optional): Device to allocate the tensor on. Defaults to "cpu".

    Returns:
        function: A function that initializes a tensor of the specified shape
        with variance scaling.

    Raises:
        ValueError: If an invalid mode or distribution is specified.

    Examples:
        # Initialize weights with normal distribution
        init_normal = variance_scaling(1.0, 'fan_in', 'normal')
        weights_normal = init_normal((64, 128))

        # Initialize weights with uniform distribution
        init_uniform = variance_scaling(1.0, 'fan_avg', 'uniform')
        weights_uniform = init_uniform((64, 128))

    Note:
        The variance scaling is a technique used to maintain the variance of
        activations across layers, which can help with training deep neural
        networks.

    Todo:
        Extend support for other distributions or modes if needed.
    """

    def _compute_fans(shape, in_axis=1, out_axis=0):
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(shape, dtype=dtype, device=device):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(
                "invalid mode for variance scaling initializer: {}".format(mode)
            )
        variance = scale / denominator
        if distribution == "normal":
            return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
        elif distribution == "uniform":
            return (
                torch.rand(*shape, dtype=dtype, device=device) * 2.0 - 1.0
            ) * np.sqrt(3 * variance)
        else:
            raise ValueError("invalid distribution for variance scaling initializer")

    return init


def default_init(scale=1.0):
    """
    Initialize weights using the same method as DDPM.

    This function returns a weight initialization function that uses
    variance scaling based on the specified scale. The initialization
    method is particularly designed to work well with Deep Denoising
    Probabilistic Models (DDPM).

    Args:
        scale (float): The scale for the variance scaling initializer.
            If scale is set to 0, it will be replaced with a small value
            (1e-10) to avoid division by zero.

    Returns:
        function: A function that takes a shape as input and returns
        initialized weights according to the variance scaling method.

    Examples:
        >>> init_func = default_init(scale=0.1)
        >>> weights = init_func((64, 128))  # Initialize weights of shape (64, 128)

    Note:
        This function is particularly useful in neural network
        architectures where weight initialization can significantly affect
        the training dynamics.
    """
    scale = 1e-10 if scale == 0 else scale
    return variance_scaling(scale, "fan_avg", "uniform")


class Dense(nn.Module):
    """
    Linear layer with `default_init`.

    This class implements a linear layer that utilizes the `default_init`
    initialization method. It is a subclass of PyTorch's `nn.Module` and
    is designed to be used in neural network architectures.

    Attributes:
        None

    Methods:
        __init__():
            Initializes the Dense layer.

    Examples:
        >>> dense_layer = Dense()
        >>> input_tensor = torch.randn(10, 20)  # Batch size of 10, 20 features
        >>> output_tensor = dense_layer(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([10, output_features])  # output_features depends on implementation
    """

    def __init__(self):
        super().__init__()


def ddpm_conv1x1(in_planes, out_planes, stride=1, bias=True, init_scale=1.0, padding=0):
    """
    1x1 convolution with DDPM initialization.

    This function creates a 1x1 convolutional layer and initializes its weights
    using the DDPM (Denoising Diffusion Probabilistic Models) method. The
    initialization is done by scaling the weights according to a specified
    scale parameter.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int, optional): Stride of the convolution. Default is 1.
        bias (bool, optional): If True, adds a learnable bias to the output.
            Default is True.
        init_scale (float, optional): Scale for weight initialization. Default
            is 1.0.
        padding (int, optional): Zero-padding added to both sides of the
            input. Default is 0.

    Returns:
        nn.Conv2d: A 1x1 convolutional layer with DDPM initialization.

    Examples:
        >>> conv_layer = ddpm_conv1x1(3, 16)
        >>> print(conv_layer)
        Conv2d(3, 16, kernel_size=(1, 1), stride=(1, 1), bias=True)

    Note:
        The weights are initialized by calling the `default_init` function
        with the specified `init_scale`. The bias is initialized to zero.
    """
    conv = nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=bias
    )
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv


def ncsn_conv3x3(
    in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1.0, padding=1
):
    """
    3x3 convolution with PyTorch initialization. Same as NCSNv1/NCSNv2.

    This function creates a 3x3 convolutional layer with the specified input and
    output channels. The layer is initialized using the given initialization scale.
    If the initialization scale is set to zero, a very small value is used to avoid
    zero initialization.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int, optional): Stride of the convolution. Default is 1.
        bias (bool, optional): If True, adds a learnable bias to the output. Default is True.
        dilation (int, optional): Spacing between kernel elements. Default is 1.
        init_scale (float, optional): Scale for initializing the weights. Default is 1.0.
        padding (int, optional): Padding added to both sides of the input. Default is 1.

    Returns:
        nn.Conv2d: A 3x3 convolutional layer with the specified parameters.

    Examples:
        >>> conv_layer = ncsn_conv3x3(in_planes=64, out_planes=128)
        >>> output = conv_layer(torch.randn(1, 64, 32, 32))
        >>> output.shape
        torch.Size([1, 128, 32, 32])

    Note:
        The convolution is designed to match the architecture of NCSNv1 and NCSNv2.
    """
    init_scale = 1e-10 if init_scale == 0 else init_scale
    conv = nn.Conv2d(
        in_planes,
        out_planes,
        stride=stride,
        bias=bias,
        dilation=dilation,
        padding=padding,
        kernel_size=3,
    )
    conv.weight.data *= init_scale
    conv.bias.data *= init_scale
    return conv


def ddpm_conv3x3(
    in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1.0, padding=1
):
    """
    3x3 convolution with DDPM initialization.

    This function creates a 3x3 convolutional layer initialized using the
    DDPM (Denoising Diffusion Probabilistic Models) method. The weights
    are initialized with a scale factor and the biases are set to zero.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        bias (bool, optional): Whether to include a bias term. Defaults to True.
        dilation (int, optional): Dilation rate for the convolution. Defaults to 1.
        init_scale (float, optional): Scale for weight initialization.
            Defaults to 1.0.
        padding (int, optional): Padding added to both sides of the input.
            Defaults to 1.

    Returns:
        nn.Conv2d: A 3x3 convolutional layer with DDPM initialization.

    Examples:
        >>> conv_layer = ddpm_conv3x3(16, 32)
        >>> output = conv_layer(torch.randn(1, 16, 64, 64))
        >>> print(output.shape)
        torch.Size([1, 32, 64, 64])

    Note:
        The weight initialization is performed using the `default_init`
        function defined elsewhere in the codebase, which applies
        variance scaling to the weights based on the given scale factor.
    """
    conv = nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
    )
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv

    ###########################################################################
    # Functions below are ported over from the NCSNv1/NCSNv2 codebase:
    # https://github.com/ermongroup/ncsn
    # https://github.com/ermongroup/ncsnv2
    ###########################################################################


class CRPBlock(nn.Module):
    """
    Channel Reduction and Pooling Block.

    This class implements a CRP (Channel Reduction and Pooling) block that
    applies a series of 3x3 convolutional layers followed by pooling to
    reduce the spatial dimensions of the input tensor while preserving
    its channel information. The block can be used as a building
    component in neural network architectures, especially in
    image processing tasks.

    Attributes:
        convs (nn.ModuleList): List of convolutional layers to apply.
        n_stages (int): Number of convolutional stages.
        pool (nn.Module): Pooling layer, either MaxPool or AvgPool.
        act (callable): Activation function applied after each convolution.

    Args:
        features (int): Number of input and output features (channels).
        n_stages (int): Number of convolutional stages to apply.
        act (callable, optional): Activation function to use (default: ReLU).
        maxpool (bool, optional): If True, uses MaxPool; else uses AvgPool
            (default: True).

    Examples:
        >>> crp_block = CRPBlock(features=64, n_stages=3)
        >>> input_tensor = torch.randn(1, 64, 32, 32)  # Batch size of 1
        >>> output_tensor = crp_block(input_tensor)
        >>> output_tensor.shape
        torch.Size([1, 64, 32, 32])

    Note:
        The input tensor is expected to have shape (N, C, H, W), where
        N is the batch size, C is the number of channels, H is the height,
        and W is the width of the input.

    Raises:
        ValueError: If `features` or `n_stages` is not positive.
    """

    def __init__(self, features, n_stages, act=nn.ReLU(), maxpool=True):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(n_stages):
            self.convs.append(ncsn_conv3x3(features, features, stride=1, bias=False))
        self.n_stages = n_stages
        if maxpool:
            self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        else:
            self.pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)

        self.act = act

    def forward(self, x):
        """
            CRPBlock module for applying convolutional layers with pooling.

        This module implements a series of convolutional layers followed by
        a pooling operation. The output of each convolutional layer is added
        to the input to form a residual connection.

        Attributes:
            convs (nn.ModuleList): List of convolutional layers.
            n_stages (int): Number of convolutional stages.
            pool (nn.Module): Pooling layer (MaxPool or AvgPool).
            act (callable): Activation function applied to the input.

        Args:
            features (int): Number of input and output features for the convolutions.
            n_stages (int): Number of convolutional stages.
            act (callable, optional): Activation function to use (default: nn.ReLU).
            maxpool (bool, optional): If True, use MaxPool; otherwise, use AvgPool
                                       (default: True).

        Returns:
            Tensor: The output tensor after applying the convolutional layers
                    and pooling operations.

        Examples:
            >>> block = CRPBlock(features=64, n_stages=3)
            >>> input_tensor = torch.randn(1, 64, 32, 32)
            >>> output_tensor = block(input_tensor)
            >>> output_tensor.shape
            torch.Size([1, 64, 32, 32])

        Note:
            The input tensor must have 4 dimensions (batch size, channels, height, width).
        """
        x = self.act(x)
        path = x
        for i in range(self.n_stages):
            path = self.pool(path)
            path = self.convs[i](path)
            x = path + x
        return x


class CondCRPBlock(nn.Module):
    """
    Conditional Residual Processing Block.

    This class implements a conditional residual processing block that applies
    a series of convolutional layers, along with conditional normalization and
    pooling operations. It is designed to facilitate enhanced feature extraction
    and integration for tasks requiring class-specific adaptations.

    Attributes:
        convs (nn.ModuleList): List of convolutional layers.
        norms (nn.ModuleList): List of normalization layers.
        normalizer (callable): Function used for normalization.
        n_stages (int): Number of stages in the block.
        pool (nn.AvgPool2d): Average pooling layer.
        act (callable): Activation function applied in the block.

    Args:
        features (int): Number of input and output features for the convolutional layers.
        n_stages (int): Number of stages (layers) to be applied in the block.
        num_classes (int): Number of classes for conditional normalization.
        normalizer (callable): Normalization function to be applied to the input.
        act (callable, optional): Activation function (default is nn.ReLU()).

    Examples:
        >>> features = 64
        >>> n_stages = 3
        >>> num_classes = 10
        >>> normalizer = ConditionalInstanceNorm2dPlus
        >>> block = CondCRPBlock(features, n_stages, num_classes, normalizer)
        >>> x = torch.randn(1, 64, 32, 32)  # Example input
        >>> y = torch.randint(0, num_classes, (1,))  # Example class input
        >>> output = block(x, y)

    Forward:
        The forward method takes an input tensor `x` and a conditional input `y`,
        applies the activation function, processes the input through the
        normalization and convolution layers, and returns the output tensor.

    Raises:
        ValueError: If the input tensor dimensions do not match the expected
        dimensions.

    Note:
        Ensure that the input tensor `x` and the conditional tensor `y` are
        appropriately shaped to match the model's expectations.
    """

    def __init__(self, features, n_stages, num_classes, normalizer, act=nn.ReLU()):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.normalizer = normalizer
        for i in range(n_stages):
            self.norms.append(normalizer(features, num_classes, bias=True))
            self.convs.append(ncsn_conv3x3(features, features, stride=1, bias=False))

        self.n_stages = n_stages
        self.pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.act = act

    def forward(self, x, y):
        """
            Forward pass through the Conditional CRP Block.

        This method applies a series of conditional normalization and convolution
        operations to the input tensor `x`, conditioned on the input tensor `y`.
        The operations are repeated over a specified number of stages, with
        intermediate pooling applied at each stage.

        Args:
            x (torch.Tensor): The input tensor of shape
                (batch_size, num_features, height, width).
            y (torch.Tensor): The conditional input tensor of shape
                (batch_size, num_classes).

        Returns:
            torch.Tensor: The output tensor of the same shape as `x`, which
            incorporates the results of the convolutional operations.

        Examples:
            >>> cond_crp_block = CondCRPBlock(features=64, n_stages=3,
            ...                                 num_classes=10,
            ...                                 normalizer=ConditionalInstanceNorm2dPlus)
            >>> x = torch.randn(8, 64, 32, 32)  # Batch of 8, 64 features
            >>> y = torch.randint(0, 10, (8,))   # Random class labels
            >>> output = cond_crp_block(x, y)
            >>> output.shape
            torch.Size([8, 64, 32, 32])
        """
        x = self.act(x)
        path = x
        for i in range(self.n_stages):
            path = self.norms[i](path, y)
            path = self.pool(path)
            path = self.convs[i](path)

            x = path + x
        return x


class RCUBlock(nn.Module):
    """
    Residual Channel Update Block.

    This block is designed to facilitate residual learning through multiple
    convolutional stages. Each stage applies a series of convolutions and
    an activation function, with the output being added back to the input
    (residual connection). This architecture helps in training deeper
    networks by mitigating the vanishing gradient problem.

    Attributes:
        n_blocks (int): The number of residual blocks to be created.
        n_stages (int): The number of convolutional stages within each block.
        act (callable): The activation function to use, default is ReLU.

    Args:
        features (int): The number of input and output channels for the
            convolutional layers.
        n_blocks (int): The number of blocks in the RCU.
        n_stages (int): The number of stages in each block.
        act (callable, optional): The activation function to apply.
            Defaults to `nn.ReLU()`.

    Example:
        >>> rcu_block = RCUBlock(features=64, n_blocks=2, n_stages=3)
        >>> input_tensor = torch.randn(1, 64, 32, 32)  # Batch of 1, 64 channels
        >>> output_tensor = rcu_block(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([1, 64, 32, 32])  # Output shape matches input shape

    Note:
        The number of input channels must match the `features` argument.
    """

    def __init__(self, features, n_blocks, n_stages, act=nn.ReLU()):
        super().__init__()

        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(
                    self,
                    "{}_{}_conv".format(i + 1, j + 1),
                    ncsn_conv3x3(features, features, stride=1, bias=False),
                )

        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.act = act

    def forward(self, x):
        """
            Residual Convolutional Unit Block (RCUBlock).

        This class implements a residual block consisting of multiple stages
        of convolutional layers. Each stage applies a specified activation
        function and an optional pooling operation to the input tensor. The
        output of each stage is added back to the input tensor, allowing for
        effective gradient flow and learning.

        Attributes:
            n_stages (int): The number of convolutional stages in the block.
            act (callable): The activation function to be applied.

        Args:
            features (int): Number of input and output features for each
                convolutional layer.
            n_blocks (int): Number of blocks to be used in the RCU.
            n_stages (int): Number of stages within each block.
            act (callable, optional): Activation function to use (default:
                nn.ReLU()).

        Examples:
            >>> rcu_block = RCUBlock(features=64, n_blocks=2, n_stages=3)
            >>> input_tensor = torch.randn(1, 64, 32, 32)
            >>> output_tensor = rcu_block(input_tensor)
            >>> output_tensor.shape
            torch.Size([1, 64, 32, 32])

        Note:
            The RCUBlock can be used as a building block for more complex
            neural network architectures, especially in tasks such as
            image processing or feature extraction in deep learning.

        Raises:
            ValueError: If n_stages or n_blocks is not a positive integer.
        """
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = self.act(x)
                x = getattr(self, "{}_{}_conv".format(i + 1, j + 1))(x)

            x += residual
        return x


class CondRCUBlock(nn.Module):
    """
    Conditional Residual Convolutional Unit Block.

    This class implements a block that consists of multiple residual convolutional
    layers, each with normalization, activation, and skip connections. It is designed
    to process input features while conditioning on an additional input (e.g., class
    labels).

    Attributes:
        n_blocks (int): The number of residual blocks in the unit.
        n_stages (int): The number of convolutional stages within each block.
        act (callable): The activation function to be used in the block.
        normalizer (callable): The normalization function to be applied to the input.

    Args:
        features (int): The number of input and output features for the convolutions.
        n_blocks (int): The number of blocks in the CondRCU.
        n_stages (int): The number of stages in each block.
        num_classes (int): The number of classes for the conditional input.
        normalizer (callable): The normalization layer to be used.
        act (callable): The activation function (default: nn.ReLU()).

    Examples:
        >>> cond_rcu = CondRCUBlock(features=64, n_blocks=2, n_stages=2,
        ...                          num_classes=10, normalizer=SomeNormalizer)
        >>> x = torch.randn(1, 64, 32, 32)  # Input tensor
        >>> y = torch.randint(0, 10, (1,))   # Conditional input (class labels)
        >>> output = cond_rcu(x, y)

    Note:
        The conditional input `y` is expected to be compatible with the normalization
        layer and should typically represent class labels or similar categorical data.

    Raises:
        ValueError: If the number of features is not positive or if the number of
        blocks or stages is not positive.
    """

    def __init__(
        self, features, n_blocks, n_stages, num_classes, normalizer, act=nn.ReLU()
    ):
        super().__init__()

        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(
                    self,
                    "{}_{}_norm".format(i + 1, j + 1),
                    normalizer(features, num_classes, bias=True),
                )
                setattr(
                    self,
                    "{}_{}_conv".format(i + 1, j + 1),
                    ncsn_conv3x3(features, features, stride=1, bias=False),
                )

        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.act = act
        self.normalizer = normalizer

    def forward(self, x, y):
        """
            Conditional Residual Unit Block.

        This class implements a conditional residual unit block which applies
        a series of convolutional layers with normalization and activation.
        The block consists of multiple stages, each containing normalization,
        convolution, and residual connections. It is particularly useful for
        tasks where conditioning on additional information (e.g., class labels)
        is required.

        Attributes:
            n_blocks (int): The number of blocks in the unit.
            n_stages (int): The number of stages in each block.
            act (callable): Activation function to be applied after normalization.
            normalizer (callable): Normalization function used for conditioning.

        Args:
            features (int): Number of input and output features for the convolutions.
            n_blocks (int): Number of residual blocks.
            n_stages (int): Number of stages in each block.
            num_classes (int): Number of classes for conditional normalization.
            normalizer (callable): Normalization layer used for conditioning.
            act (callable, optional): Activation function (default is nn.ReLU()).

        Forward Method:
            The forward method takes two inputs:
                x (torch.Tensor): The input tensor of shape (B, C, H, W).
                y (torch.Tensor): The conditioning tensor of shape (B, num_classes).

            Returns:
                torch.Tensor: The output tensor of shape (B, C, H, W) after
                applying the conditional residual block.

        Examples:
            >>> cond_rcu_block = CondRCUBlock(features=64, n_blocks=2,
            ...                                 n_stages=2, num_classes=10,
            ...                                 normalizer=some_normalizer)
            >>> x = torch.randn(8, 64, 32, 32)  # Example input
            >>> y = torch.randint(0, 10, (8,))   # Example conditioning input
            >>> output = cond_rcu_block(x, y)
            >>> print(output.shape)
            torch.Size([8, 64, 32, 32])

        Note:
            This block is particularly effective in generative models where
            conditioning can significantly improve performance.

        Todo:
            - Implement additional normalization options.
            - Optimize the block for performance on large models.
        """
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = getattr(self, "{}_{}_norm".format(i + 1, j + 1))(x, y)
                x = self.act(x)
                x = getattr(self, "{}_{}_conv".format(i + 1, j + 1))(x)

            x += residual
        return x


class MSFBlock(nn.Module):
    """
    Multi-Scale Fusion Block for feature aggregation across multiple scales.

    This block is designed to take multiple input feature maps of different
    channels and aggregate them into a single output feature map with a
    specified number of output features. It applies a 3x3 convolution to
    each input feature map, followed by an interpolation to a common output
    shape, and sums the results.

    Attributes:
        convs (nn.ModuleList): A list of convolutional layers for each input
            feature map.
        features (int): The number of output features for the block.

    Args:
        in_planes (list or tuple): A list or tuple of integers representing the
            number of input channels for each feature map.
        features (int): The number of output channels after the convolution.

    Returns:
        torch.Tensor: A tensor containing the aggregated output feature map.

    Examples:
        >>> msf_block = MSFBlock([64, 128, 256], features=128)
        >>> input_features = [torch.randn(1, 64, 32, 32),
        ...                   torch.randn(1, 128, 16, 16),
        ...                   torch.randn(1, 256, 8, 8)]
        >>> output = msf_block(input_features, shape=(32, 32))
        >>> output.shape
        torch.Size([1, 128, 32, 32])

    Note:
        The input feature maps should be provided as a list or tuple, and
        all feature maps should be of the same batch size.

    Raises:
        AssertionError: If the input `in_planes` is not a list or tuple.
    """

    def __init__(self, in_planes, features):
        super().__init__()
        assert isinstance(in_planes, list) or isinstance(in_planes, tuple)
        self.convs = nn.ModuleList()
        self.features = features

        for i in range(len(in_planes)):
            self.convs.append(ncsn_conv3x3(in_planes[i], features, stride=1, bias=True))

    def forward(self, xs, shape):
        """
            Multi-Scale Feature Block for processing multiple input feature maps.

        This block takes a list of input feature maps and applies 3x3 convolutions
        to each one, followed by bilinear interpolation to combine them into a
        single output feature map.

        Attributes:
            convs (nn.ModuleList): A list of convolutional layers for each input
                feature map.
            features (int): The number of output feature channels.

        Args:
            in_planes (list or tuple): A list or tuple of integers representing
                the number of input channels for each feature map.
            features (int): The number of output feature channels.

        Returns:
            torch.Tensor: A tensor containing the combined output feature map.

        Examples:
            >>> msf_block = MSFBlock([64, 128], features=256)
            >>> input_tensor1 = torch.randn(1, 64, 32, 32)
            >>> input_tensor2 = torch.randn(1, 128, 32, 32)
            >>> output = msf_block([input_tensor1, input_tensor2], shape=(64, 64))
            >>> output.shape
            torch.Size([1, 256, 64, 64])
        """
        sums = torch.zeros(xs[0].shape[0], self.features, *shape, device=xs[0].device)
        for i in range(len(self.convs)):
            h = self.convs[i](xs[i])
            h = F.interpolate(h, size=shape, mode="bilinear", align_corners=True)
            sums += h
        return sums


class CondMSFBlock(nn.Module):
    """
    Conditional Multi-Scale Fusion Block.

    This block is designed for multi-scale feature fusion in a conditional
    manner, typically used in tasks such as image generation and enhancement.
    It combines multiple input feature maps using convolutions and normalizes
    them based on class information.

    Attributes:
        convs (nn.ModuleList): List of convolutional layers for each input plane.
        norms (nn.ModuleList): List of normalization layers for each input plane.
        features (int): The number of output features for the block.
        normalizer (callable): A callable normalization function.

    Args:
        in_planes (list or tuple): A list or tuple containing the number of input
            channels for each input feature map.
        features (int): The number of output features after fusion.
        num_classes (int): The number of classes for conditional normalization.
        normalizer (callable): A normalization layer or function to be applied
            after convolutions.

    Forward:
        x (list of tensors): List of input feature maps, each of shape
            (B, C, H, W), where B is the batch size, C is the number of
            channels, H is the height, and W is the width.
        y (tensor): A tensor containing class information, used for
            conditional normalization.
        shape (tuple): The target shape (H, W) for the output feature map.

    Returns:
        tensor: A tensor of shape (B, features, height, width) containing
        the fused feature map.

    Examples:
        >>> cond_msf = CondMSFBlock([32, 64], features=128, num_classes=10,
        ...                          normalizer=ConditionalInstanceNorm2dPlus)
        >>> input_features = [torch.randn(8, 32, 64, 64), torch.randn(8, 64, 64, 64)]
        >>> class_info = torch.randint(0, 10, (8,))
        >>> output = cond_msf(input_features, class_info, (32, 32))
        >>> output.shape
        torch.Size([8, 128, 32, 32])

    Note:
        The normalization layers expect the input tensors to be of shape
        (B, C, H, W). Ensure that the input tensors are appropriately shaped
        before passing them to the forward method.

    Todo:
        Add support for additional normalization techniques in the future.
    """

    def __init__(self, in_planes, features, num_classes, normalizer):
        super().__init__()
        assert isinstance(in_planes, list) or isinstance(in_planes, tuple)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.features = features
        self.normalizer = normalizer

        for i in range(len(in_planes)):
            self.convs.append(ncsn_conv3x3(in_planes[i], features, stride=1, bias=True))
            self.norms.append(normalizer(in_planes[i], num_classes, bias=True))

    def forward(self, xs, y, shape):
        """
            Performs the forward pass of the Conditional Residual Block.

        This method takes the input tensor `x` and the conditioning tensor `y`,
        applies normalization, activation, and convolution operations in a
        residual manner across multiple blocks and stages. The output is a
        combination of the processed input and the residual.

        Args:
            x (torch.Tensor): The input tensor of shape (B, C, H, W), where
                B is the batch size, C is the number of channels, and
                H and W are the height and width of the input.
            y (torch.Tensor): The conditioning tensor of shape (B, num_classes),
                where B is the batch size and num_classes is the number of classes
                for conditional normalization.

        Returns:
            torch.Tensor: The output tensor after applying the forward operations,
                with the same shape as the input tensor `x`.

        Examples:
            >>> block = CondRCUBlock(features=64, n_blocks=2, n_stages=2,
            ...                       num_classes=10, normalizer=SomeNormalizer)
            >>> x = torch.randn(8, 64, 32, 32)  # Batch of 8, 64 channels, 32x32
            >>> y = torch.randint(0, 10, (8,))   # Batch of 8, class indices
            >>> output = block(x, y)
            >>> print(output.shape)
            torch.Size([8, 64, 32, 32])

        Note:
            This block uses conditional normalization, which allows the
            model to adapt its parameters based on the class information
            provided by the conditioning tensor `y`.

        Raises:
            RuntimeError: If the shapes of the input tensor `x` and
                conditioning tensor `y` do not match the expected dimensions.
        """
        sums = torch.zeros(xs[0].shape[0], self.features, *shape, device=xs[0].device)
        for i in range(len(self.convs)):
            h = self.norms[i](xs[i], y)
            h = self.convs[i](h)
            h = F.interpolate(h, size=shape, mode="bilinear", align_corners=True)
            sums += h
        return sums


class RefineBlock(nn.Module):
    """
    Refinement block for feature extraction and processing.

    This block is designed to adaptively refine features from multiple
    input sources using Residual Convolutional Units (RCU), a Multi-Scale
    Feature (MSF) block, and a Conditional Residual Processing (CRP)
    block. It allows for the combination of features from different
    resolutions and enhances them through a series of convolutions.

    Attributes:
        adapt_convs (nn.ModuleList): List of RCU blocks for adapting input features.
        output_convs (RCUBlock): RCU block for generating output features.
        msf (MSFBlock): Multi-Scale Feature block for combining features.
        crp (CRPBlock): Conditional Residual Processing block for refining features.

    Args:
        in_planes (tuple or list): Number of input feature planes for each block.
        features (int): Number of output feature planes for the final output.
        act (callable): Activation function to use (default: ReLU).
        start (bool): Flag indicating if this block is the starting block (default: False).
        end (bool): Flag indicating if this block is the ending block (default: False).
        maxpool (bool): Flag indicating if max pooling should be used (default: True).

    Raises:
        AssertionError: If `in_planes` is not a tuple or list.

    Examples:
        >>> refine_block = RefineBlock((64, 128), 256)
        >>> output = refine_block((input_tensor1, input_tensor2), output_shape)

    Note:
        The `start` and `end` flags can be used to customize the behavior
        of the block based on its position in the overall network architecture.
    """

    def __init__(
        self, in_planes, features, act=nn.ReLU(), start=False, end=False, maxpool=True
    ):
        super().__init__()

        assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
        self.n_blocks = n_blocks = len(in_planes)

        self.adapt_convs = nn.ModuleList()
        for i in range(n_blocks):
            self.adapt_convs.append(RCUBlock(in_planes[i], 2, 2, act))

        self.output_convs = RCUBlock(features, 3 if end else 1, 2, act)

        if not start:
            self.msf = MSFBlock(in_planes, features)

        self.crp = CRPBlock(features, 2, act, maxpool=maxpool)

    def forward(self, xs, output_shape):
        """
            RefineBlock for hierarchical feature refinement in deep networks.

        This module refines input features from multiple sources through a series of
        convolutional layers, pooling operations, and residual connections. It is designed
        to adaptively fuse features from various input resolutions, enhancing the network's
        ability to capture complex patterns.

        Attributes:
            n_blocks (int): Number of input feature blocks.
            adapt_convs (nn.ModuleList): List of Residual Convolutional Units (RCUs)
                for adapting input features.
            output_convs (RCUBlock): RCU for producing output features.
            msf (MSFBlock): Multi-scale feature fusion block, if applicable.
            crp (CRPBlock): Contextual Refinement Pooling block.

        Args:
            in_planes (list or tuple): A list or tuple of input feature dimensions.
            features (int): Number of output features for the final layer.
            act (callable): Activation function to use (default: nn.ReLU).
            start (bool): Whether this block is at the start of the network (default: False).
            end (bool): Whether this block is at the end of the network (default: False).
            maxpool (bool): Whether to use max pooling in CRPBlock (default: True).

        Examples:
            >>> refine_block = RefineBlock(in_planes=(64, 128), features=256)
            >>> input_features = (torch.randn(1, 64, 32, 32), torch.randn(1, 128, 16, 16))
            >>> output = refine_block(input_features, output_shape=(16, 16))
            >>> print(output.shape)
            torch.Size([1, 256, 16, 16])
        """
        assert isinstance(xs, tuple) or isinstance(xs, list)
        hs = []
        for i in range(len(xs)):
            h = self.adapt_convs[i](xs[i])
            hs.append(h)

        if self.n_blocks > 1:
            h = self.msf(hs, output_shape)
        else:
            h = hs[0]

        h = self.crp(h)
        h = self.output_convs(h)

        return h


class CondRefineBlock(nn.Module):
    """
    Conditional Refine Block for processing input features with class conditioning.

    This block applies a series of conditional residual units and pooling layers
    to refine the input features while taking class information into account.
    It is particularly useful in tasks where the output needs to be conditioned
    on class labels, enhancing the feature representation based on the classes.

    Attributes:
        adapt_convs (nn.ModuleList): A list of conditional residual blocks for
            adapting input features.
        output_convs (CondRCUBlock): A conditional residual block for producing
            the output features.
        msf (CondMSFBlock): A multi-scale feature block to combine features from
            multiple input sources.
        crp (CondCRPBlock): A conditional CRP block for pooling and refining
            features.

    Args:
        in_planes (tuple or list): Number of input channels for each block.
        features (int): Number of output channels for the final output.
        num_classes (int): Number of classes for conditional normalization.
        normalizer (callable): Normalization function to be applied.
        act (nn.Module, optional): Activation function to use. Defaults to nn.ReLU().
        start (bool, optional): Flag indicating if this is the start block.
            Defaults to False.
        end (bool, optional): Flag indicating if this is the end block.
            Defaults to False.

    Examples:
        >>> cond_refine_block = CondRefineBlock(
        ...     in_planes=(32, 64),
        ...     features=128,
        ...     num_classes=10,
        ...     normalizer=ConditionalInstanceNorm2dPlus
        ... )
        >>> x = (torch.randn(1, 32, 64, 64), torch.randn(1, 64, 32, 32))
        >>> y = torch.randint(0, 10, (1,))
        >>> output = cond_refine_block(x, y, output_shape=(32, 32))

    Note:
        The input `xs` should be a tuple or list of feature maps to be processed,
        and `y` should be a tensor containing the class labels.

    Raises:
        AssertionError: If `xs` is not a tuple or list.
    """

    def __init__(
        self,
        in_planes,
        features,
        num_classes,
        normalizer,
        act=nn.ReLU(),
        start=False,
        end=False,
    ):
        super().__init__()

        assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
        self.n_blocks = n_blocks = len(in_planes)

        self.adapt_convs = nn.ModuleList()
        for i in range(n_blocks):
            self.adapt_convs.append(
                CondRCUBlock(in_planes[i], 2, 2, num_classes, normalizer, act)
            )

        self.output_convs = CondRCUBlock(
            features, 3 if end else 1, 2, num_classes, normalizer, act
        )

        if not start:
            self.msf = CondMSFBlock(in_planes, features, num_classes, normalizer)

        self.crp = CondCRPBlock(features, 2, num_classes, normalizer, act)

    def forward(self, xs, y, output_shape):
        """
            Conditional Refine Block for enhancing feature representations.

        This block is designed to refine features through a combination of
        conditional residual units and pooling layers. It utilizes multiple
        stages of processing, where each stage applies normalization,
        convolutions, and an activation function. The block is particularly
        useful in conditional settings, allowing the integration of class
        information during the refinement process.

        Attributes:
            n_blocks (int): Number of input blocks.
            adapt_convs (nn.ModuleList): List of conditional residual units
                for each input block.
            output_convs (CondRCUBlock): Conditional residual unit for output
                features.
            msf (CondMSFBlock): Multi-scale feature block for combining inputs
                (if not starting).
            crp (CondCRPBlock): Conditional Residual Pooling block for refining
                features.

        Args:
            in_planes (tuple or list): Number of input feature planes for
                each block.
            features (int): Number of feature planes for output.
            num_classes (int): Number of classes for conditional normalization.
            normalizer (callable): Normalization function to be used in the
                conditional residual units.
            act (nn.Module, optional): Activation function to apply. Defaults
                to ReLU.
            start (bool, optional): If True, skip multi-scale feature block.
                Defaults to False.
            end (bool, optional): If True, apply additional layers at the end.
                Defaults to False.

        Returns:
            Tensor: Refined feature tensor after processing.

        Examples:
            >>> cond_refine_block = CondRefineBlock(
            ...     in_planes=(32, 64),
            ...     features=128,
            ...     num_classes=10,
            ...     normalizer=ConditionalInstanceNorm2dPlus
            ... )
            >>> x = (torch.randn(1, 32, 64, 64), torch.randn(1, 64, 64, 64))
            >>> y = torch.randint(0, 10, (1,))
            >>> output_shape = (128, 32, 32)
            >>> output = cond_refine_block(x, y, output_shape)
            >>> output.shape
            torch.Size([1, 128, 32, 32])

        Note:
            Ensure that the input tensors are compatible with the expected
            dimensions, particularly with respect to the number of channels
            and spatial dimensions.

        Todo:
            - Add support for additional normalization techniques.
            - Explore different activation functions to enhance performance.
        """
        assert isinstance(xs, tuple) or isinstance(xs, list)
        hs = []
        for i in range(len(xs)):
            h = self.adapt_convs[i](xs[i], y)
            hs.append(h)

        if self.n_blocks > 1:
            h = self.msf(hs, y, output_shape)
        else:
            h = hs[0]

        h = self.crp(h, y)
        h = self.output_convs(h, y)

        return h


class ConvMeanPool(nn.Module):
    """
    Convolutional layer followed by mean pooling.

    This layer applies a convolution operation followed by mean pooling
    on the input tensor. The mean pooling is done over the four quadrants
    of the input, effectively downsampling the feature map while
    preserving the learned features.

    Attributes:
        conv (nn.Module): The convolutional layer used to process the input.

    Args:
        input_dim (int): The number of input channels.
        output_dim (int): The number of output channels.
        kernel_size (int, optional): The size of the convolutional kernel.
            Defaults to 3.
        biases (bool, optional): Whether to include a bias term in the
            convolution. Defaults to True.
        adjust_padding (bool, optional): If True, applies zero padding
            before the convolution to maintain the spatial dimensions.
            Defaults to False.

    Examples:
        >>> layer = ConvMeanPool(input_dim=3, output_dim=16, kernel_size=3)
        >>> input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1
        >>> output_tensor = layer(input_tensor)
        >>> output_tensor.shape
        torch.Size([1, 16, 16, 16])  # Output dimensions after pooling

    Note:
        If `adjust_padding` is set to True, the convolution will be applied
        with an additional padding of 1 pixel on the left and top sides.
    """

    def __init__(
        self, input_dim, output_dim, kernel_size=3, biases=True, adjust_padding=False
    ):
        super().__init__()
        if not adjust_padding:
            conv = nn.Conv2d(
                input_dim,
                output_dim,
                kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=biases,
            )
            self.conv = conv
        else:
            conv = nn.Conv2d(
                input_dim,
                output_dim,
                kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=biases,
            )

            self.conv = nn.Sequential(nn.ZeroPad2d((1, 0, 1, 0)), conv)

    def forward(self, inputs):
        """
            Forward pass for the Conditional Residual Block.

        This method applies the forward operation on the input tensor `x`
        and the conditional tensor `y`. It normalizes the input, applies
        a series of convolutions, and returns the sum of the shortcut
        connection and the output of the convolutions.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W), where
                B is the batch size, C is the number of channels,
                H is the height, and W is the width.
            y (torch.Tensor): Conditional tensor of shape (B, num_classes).

        Returns:
            torch.Tensor: Output tensor of the same shape as input `x`.

        Note:
            The number of channels in `x` must match `input_dim`,
            and the number of classes in `y` must match `num_classes`.

        Examples:
            >>> block = ConditionalResidualBlock(input_dim=64, output_dim=128,
            ...                                   num_classes=10)
            >>> x = torch.randn(8, 64, 32, 32)  # Batch of 8 images
            >>> y = torch.randint(0, 10, (8,))  # Batch of class labels
            >>> output = block(x, y)
            >>> print(output.shape)
            torch.Size([8, 128, 32, 32])
        """
        output = self.conv(inputs)
        output = (
            sum(
                [
                    output[:, :, ::2, ::2],
                    output[:, :, 1::2, ::2],
                    output[:, :, ::2, 1::2],
                    output[:, :, 1::2, 1::2],
                ]
            )
            / 4.0
        )
        return output


class MeanPoolConv(nn.Module):
    """
    Mean Pooling followed by a Convolutional layer.

    This class applies a mean pooling operation on the input tensor, followed
    by a convolution operation. The mean pooling is performed over a 2x2
    spatial region, which effectively reduces the spatial dimensions by half.

    Attributes:
        conv (nn.Conv2d): The convolutional layer that processes the output
            from the mean pooling operation.

    Args:
        input_dim (int): The number of input channels.
        output_dim (int): The number of output channels.
        kernel_size (int, optional): The size of the convolutional kernel.
            Default is 3.
        biases (bool, optional): If True, adds a learnable bias to the
            convolutional layer. Default is True.

    Returns:
        Tensor: The output of the convolution after applying mean pooling
        on the input.

    Examples:
        >>> mean_pool_conv = MeanPoolConv(input_dim=3, output_dim=16)
        >>> input_tensor = torch.randn(1, 3, 64, 64)  # (batch_size, channels, height, width)
        >>> output_tensor = mean_pool_conv(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([1, 16, 32, 32])  # Output shape after mean pooling and convolution

    Note:
        The mean pooling operation averages the values in a 2x2 region
        across the input tensor, which reduces the spatial dimensions by
        a factor of 2. The convolutional layer then processes this pooled
        output to produce the final result.
    """

    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True):
        super().__init__()
        self.conv = nn.Conv2d(
            input_dim,
            output_dim,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=biases,
        )

    def forward(self, inputs):
        """
            Performs the forward pass of the Conditional Residual Block.

        This method takes an input tensor and a conditional tensor, processes them
        through several convolutional layers, applies normalization and activation
        functions, and finally returns the output of the residual connection.

        Args:
            x (torch.Tensor): The input tensor of shape (B, C, H, W), where:
                B = batch size
                C = number of input channels
                H = height of the input tensor
                W = width of the input tensor
            y (torch.Tensor): The conditional tensor of shape (B, num_classes),
                which is used for conditional normalization.

        Returns:
            torch.Tensor: The output tensor of the same shape as the input tensor
            after applying the conditional residual operations.

        Examples:
            >>> block = ConditionalResidualBlock(input_dim=64, output_dim=128,
            ...                                   num_classes=10)
            >>> input_tensor = torch.randn(8, 64, 32, 32)  # Batch of 8 images
            >>> conditional_tensor = torch.randint(0, 10, (8, 10))  # Batch of labels
            >>> output = block(input_tensor, conditional_tensor)
            >>> print(output.shape)  # Output shape should be (8, 128, 32, 32)

        Note:
            The class should be initialized with the appropriate dimensions and
            normalization method to ensure proper functionality.

        Raises:
            Exception: If the output dimension does not match the input dimension
            and resampling is not None.
        """
        output = inputs
        output = (
            sum(
                [
                    output[:, :, ::2, ::2],
                    output[:, :, 1::2, ::2],
                    output[:, :, ::2, 1::2],
                    output[:, :, 1::2, 1::2],
                ]
            )
            / 4.0
        )
        return self.conv(output)


class UpsampleConv(nn.Module):
    """
    Upsampling convolution layer.

    This layer performs upsampling on the input tensor using PixelShuffle
    after concatenating the input with itself four times. The upsampled
    output is then passed through a convolutional layer.

    Attributes:
        conv (nn.Conv2d): The convolutional layer that processes the output
            after upsampling.
        pixelshuffle (nn.PixelShuffle): The layer that performs pixel shuffling
            to upscale the input.

    Args:
        input_dim (int): Number of input channels.
        output_dim (int): Number of output channels.
        kernel_size (int, optional): Size of the convolving kernel. Default is 3.
        biases (bool, optional): If True, adds a learnable bias to the output.
            Default is True.

    Returns:
        Tensor: The output tensor after upsampling and convolution.

    Examples:
        >>> upsample_conv = UpsampleConv(input_dim=16, output_dim=32)
        >>> input_tensor = torch.randn(1, 16, 64, 64)
        >>> output_tensor = upsample_conv(input_tensor)
        >>> output_tensor.shape
        torch.Size([1, 32, 128, 128])

    Note:
        The input tensor is expected to have 4 dimensions: (batch_size,
        channels, height, width). The output tensor will have its height and
        width doubled.

    Todo:
        - Consider adding options for different upsampling methods in the
          future.
    """

    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True):
        super().__init__()
        self.conv = nn.Conv2d(
            input_dim,
            output_dim,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=biases,
        )
        self.pixelshuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, inputs):
        """
            Forward pass through the Conditional Residual Block.

        This method applies a series of normalization, activation, and convolutional
        operations to the input tensor `x`. It also incorporates a conditional
        input `y` for normalization purposes, allowing the block to learn
        class-specific features.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W), where B is the
                batch size, C is the number of input channels, H is the height,
                and W is the width of the input.
            y (torch.Tensor): Conditional input tensor of shape (B, num_classes)
                used for normalization.

        Returns:
            torch.Tensor: Output tensor of the same shape as input `x`,
                after applying the operations defined in the block.

        Note:
            The behavior of the block depends on the `resample` attribute:
            - If `resample` is "down", the spatial dimensions of the output
              are reduced.
            - If `resample` is None, the input dimensions remain unchanged.
            - If the output dimensions differ from the input dimensions, a
              shortcut connection is applied to match the dimensions.

        Examples:
            >>> block = ConditionalResidualBlock(input_dim=64, output_dim=128,
            ...                                   num_classes=10)
            >>> x = torch.randn(8, 64, 32, 32)  # Batch of 8, 64 channels, 32x32
            >>> y = torch.randint(0, 10, (8,))   # Random class labels for 8 samples
            >>> output = block(x, y)
            >>> output.shape
            torch.Size([8, 128, 32, 32])  # Output shape will be (B, output_dim, H, W)
        """
        output = inputs
        output = torch.cat([output, output, output, output], dim=1)
        output = self.pixelshuffle(output)
        return self.conv(output)


class ConditionalResidualBlock(nn.Module):
    """
    Conditional Residual Block for neural network architectures.

    This block implements a residual connection with conditional normalization.
    It can be used in various architectures where residual connections and
    conditioning based on class labels are needed. The block supports
    downsampling, dilation, and different types of activation functions.

    Attributes:
        non_linearity (callable): Activation function to apply.
        input_dim (int): Number of input channels.
        output_dim (int): Number of output channels.
        resample (Union[int, None]): Resampling method. Can be 'down', None, or an int.
        normalization (callable): Normalization method to use for the layers.
        conv1 (nn.Module): First convolutional layer.
        normalize1 (nn.Module): First normalization layer.
        normalize2 (nn.Module): Second normalization layer.
        conv2 (nn.Module): Second convolutional layer.
        shortcut (nn.Module): Shortcut connection layer if needed.

    Args:
        input_dim (int): Number of input channels.
        output_dim (int): Number of output channels.
        num_classes (int): Number of classes for conditioning.
        resample (Union[str, None]): Resampling method ('down', None).
        act (callable): Activation function (default: nn.ELU()).
        normalization (callable): Normalization layer (default: ConditionalInstanceNorm2dPlus).
        adjust_padding (bool): Whether to adjust padding (default: False).
        dilation (int): Dilation rate for convolutions (default: None).

    Returns:
        Tensor: The output tensor after applying the block.

    Examples:
        >>> block = ConditionalResidualBlock(input_dim=64, output_dim=128,
        ...                                   num_classes=10, resample='down')
        >>> x = torch.randn(16, 64, 32, 32)  # Batch of 16, 64 channels, 32x32 size
        >>> y = torch.randint(0, 10, (16,))   # Random class labels for conditioning
        >>> output = block(x, y)
        >>> output.shape
        torch.Size([16, 128, 16, 16])  # Output shape after downsampling

    Note:
        The block assumes that the input tensor is in the format (N, C, H, W),
        where N is the batch size, C is the number of channels, and H and W are
        the height and width of the input feature maps.

    Raises:
        Exception: If an invalid resample value is provided.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_classes,
        resample=1,
        act=nn.ELU(),
        normalization=ConditionalInstanceNorm2dPlus,
        adjust_padding=False,
        dilation=None,
    ):
        super().__init__()
        self.non_linearity = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        self.normalization = normalization
        if resample == "down":
            if dilation > 1:
                self.conv1 = ncsn_conv3x3(input_dim, input_dim, dilation=dilation)
                self.normalize2 = normalization(input_dim, num_classes)
                self.conv2 = ncsn_conv3x3(input_dim, output_dim, dilation=dilation)
                conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
            else:
                self.conv1 = ncsn_conv3x3(input_dim, input_dim)
                self.normalize2 = normalization(input_dim, num_classes)
                self.conv2 = ConvMeanPool(
                    input_dim, output_dim, 3, adjust_padding=adjust_padding
                )
                conv_shortcut = partial(
                    ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding
                )

        elif resample is None:
            if dilation > 1:
                conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
                self.conv1 = ncsn_conv3x3(input_dim, output_dim, dilation=dilation)
                self.normalize2 = normalization(output_dim, num_classes)
                self.conv2 = ncsn_conv3x3(output_dim, output_dim, dilation=dilation)
            else:
                conv_shortcut = nn.Conv2d
                self.conv1 = ncsn_conv3x3(input_dim, output_dim)
                self.normalize2 = normalization(output_dim, num_classes)
                self.conv2 = ncsn_conv3x3(output_dim, output_dim)
        else:
            raise Exception("invalid resample value")

        if output_dim != input_dim or resample is not None:
            self.shortcut = conv_shortcut(input_dim, output_dim)

        self.normalize1 = normalization(input_dim, num_classes)

    def forward(self, x, y):
        """
            Conditional Residual Block for neural network architectures.

        This block implements a residual connection with conditional normalization,
        which is useful in generative models, particularly in diffusion models.
        The block can perform downsampling, maintain spatial dimensions, or
        upsample based on the specified configuration. It uses activation functions
        and normalization layers to enhance learning and representation.

        Attributes:
            non_linearity (nn.Module): The activation function to apply.
            input_dim (int): The number of input channels.
            output_dim (int): The number of output channels.
            resample (Union[int, str, None]): The resampling strategy.
            normalization (callable): The normalization layer to apply.
            conv1 (nn.Module): The first convolutional layer.
            normalize1 (nn.Module): The first normalization layer.
            normalize2 (nn.Module): The second normalization layer.
            conv2 (nn.Module): The second convolutional layer.
            shortcut (nn.Module): The shortcut connection for residual learning.

        Args:
            input_dim (int): Number of input channels.
            output_dim (int): Number of output channels.
            num_classes (int): Number of classes for conditional normalization.
            resample (Union[int, str, None]): Resampling strategy ('down', None).
            act (nn.Module): Activation function to use (default: nn.ELU()).
            normalization (callable): Normalization layer to use (default:
                ConditionalInstanceNorm2dPlus).
            adjust_padding (bool): Whether to adjust padding (default: False).
            dilation (Optional[int]): Dilation rate for convolution (default: None).

        Returns:
            torch.Tensor: The output tensor after applying the residual block.

        Examples:
            >>> block = ConditionalResidualBlock(input_dim=64, output_dim=128,
            ...                                   num_classes=10, resample='down')
            >>> x = torch.randn(8, 64, 32, 32)  # Batch of 8 images
            >>> y = torch.randint(0, 10, (8,))  # Batch of 8 class labels
            >>> output = block(x, y)
            >>> print(output.shape)  # Output shape should be (8, 128, 16, 16)

        Raises:
            Exception: If an invalid resample value is provided.
        """
        output = self.normalize1(x, y)
        output = self.non_linearity(output)
        output = self.conv1(output)
        output = self.normalize2(output, y)
        output = self.non_linearity(output)
        output = self.conv2(output)

        if self.output_dim == self.input_dim and self.resample is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)

        return shortcut + output


class ResidualBlock(nn.Module):
    """
    Residual Block for deep learning architectures.

    This class implements a residual block, which consists of two convolutional
    layers with a skip connection. The block can perform downsampling and apply
    normalization and activation functions as specified in the initialization.

    Attributes:
        non_linearity (callable): The activation function to apply.
        input_dim (int): The number of input channels.
        output_dim (int): The number of output channels.
        resample (str or None): If 'down', the block will downsample the input.
        normalization (callable): The normalization layer to use.
        shortcut (nn.Module): The shortcut connection layer.
        normalize1 (callable): The normalization layer applied before the first
            convolution.
        normalize2 (callable): The normalization layer applied after the first
            convolution.

    Args:
        input_dim (int): Number of input channels.
        output_dim (int): Number of output channels.
        resample (str, optional): 'down' for downsampling, None for no change.
        act (callable, optional): Activation function, default is nn.ELU().
        normalization (callable, optional): Normalization layer, default is
            nn.InstanceNorm2d.
        adjust_padding (bool, optional): If True, adjusts padding for convolutions.
        dilation (int, optional): Dilation rate for convolutions, default is 1.

    Returns:
        Tensor: The output tensor after applying the residual block.

    Raises:
        Exception: If an invalid resample value is provided.

    Examples:
        >>> block = ResidualBlock(input_dim=64, output_dim=128, resample='down')
        >>> input_tensor = torch.randn(1, 64, 32, 32)
        >>> output_tensor = block(input_tensor)
        >>> output_tensor.shape
        torch.Size([1, 128, 16, 16])
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        resample=None,
        act=nn.ELU(),
        normalization=nn.InstanceNorm2d,
        adjust_padding=False,
        dilation=1,
    ):
        super().__init__()
        self.non_linearity = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        self.normalization = normalization
        if resample == "down":
            if dilation > 1:
                self.conv1 = ncsn_conv3x3(input_dim, input_dim, dilation=dilation)
                self.normalize2 = normalization(input_dim)
                self.conv2 = ncsn_conv3x3(input_dim, output_dim, dilation=dilation)
                conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
            else:
                self.conv1 = ncsn_conv3x3(input_dim, input_dim)
                self.normalize2 = normalization(input_dim)
                self.conv2 = ConvMeanPool(
                    input_dim, output_dim, 3, adjust_padding=adjust_padding
                )
                conv_shortcut = partial(
                    ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding
                )

        elif resample is None:
            if dilation > 1:
                conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
                self.conv1 = ncsn_conv3x3(input_dim, output_dim, dilation=dilation)
                self.normalize2 = normalization(output_dim)
                self.conv2 = ncsn_conv3x3(output_dim, output_dim, dilation=dilation)
            else:
                # conv_shortcut = nn.Conv2d ### Something wierd here.
                conv_shortcut = partial(ncsn_conv1x1)
                self.conv1 = ncsn_conv3x3(input_dim, output_dim)
                self.normalize2 = normalization(output_dim)
                self.conv2 = ncsn_conv3x3(output_dim, output_dim)
        else:
            raise Exception("invalid resample value")

        if output_dim != input_dim or resample is not None:
            self.shortcut = conv_shortcut(input_dim, output_dim)

        self.normalize1 = normalization(input_dim)

    def forward(self, x):
        """
            Perform the forward pass of the Residual Block.

        This method applies the forward operations of the Residual Block,
        which includes normalization, non-linearity, convolutional operations,
        and adding the shortcut connection.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W), where:
                B = batch size,
                C = number of input channels,
                H = height of the input feature map,
                W = width of the input feature map.

        Returns:
            torch.Tensor: Output tensor of the same shape as the input tensor.

        Note:
            The output of the block is computed as the sum of the input
            tensor and the processed tensor through the block. If the
            dimensions of the input and output tensors do not match,
            a convolution is applied to the input tensor to adjust its
            dimensions before summation.

        Examples:
            >>> block = ResidualBlock(input_dim=64, output_dim=128)
            >>> input_tensor = torch.randn(32, 64, 128, 128)
            >>> output_tensor = block(input_tensor)
            >>> print(output_tensor.shape)
            torch.Size([32, 128, 128, 128])
        """
        output = self.normalize1(x)
        output = self.non_linearity(output)
        output = self.conv1(output)
        output = self.normalize2(output)
        output = self.non_linearity(output)
        output = self.conv2(output)

        if self.output_dim == self.input_dim and self.resample is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)

        return shortcut + output


###########################################################################
# Functions below are ported over from the DDPM codebase:
#  https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
###########################################################################


def _einsum(a, b, c, x, y):
    einsum_str = "{},{}->{}".format("".join(a), "".join(b), "".join(c))
    return torch.einsum(einsum_str, x, y)


def contract_inner(x, y):
    """
    Compute the inner product of two tensors along the last axis of x
    and the first axis of y.

    This function utilizes Einstein summation convention to compute the
    inner product, effectively performing a tensordot operation with
    the specified axes.

    Args:
        x (torch.Tensor): The first input tensor of shape (N, ..., M).
        y (torch.Tensor): The second input tensor of shape (N, ..., P).

    Returns:
        torch.Tensor: The result of the inner product of x and y,
        with shape (N, ..., P).

    Examples:
        >>> x = torch.randn(2, 3, 4)
        >>> y = torch.randn(2, 4, 5)
        >>> result = contract_inner(x, y)
        >>> result.shape
        torch.Size([2, 3, 5])

    Note:
        - The function assumes that the last dimension of `x` and the first
          dimension of `y` are compatible for the inner product.
    """
    x_chars = list(string.ascii_lowercase[: len(x.shape)])
    y_chars = list(string.ascii_lowercase[len(x.shape) : len(y.shape) + len(x.shape)])
    y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
    out_chars = x_chars[:-1] + y_chars[1:]
    return _einsum(x_chars, y_chars, out_chars, x, y)


class NIN(nn.Module):
    """
    NIN (Network in Network) layer.

    This layer applies a Network in Network (NIN) operation, which is a form
    of deep learning layer designed to learn more complex features by using
    multiple small convolutional layers instead of traditional convolutional
    layers.

    Attributes:
        W (torch.nn.Parameter): Weight parameter of the layer, initialized
            using a default scale.
        b (torch.nn.Parameter): Bias parameter of the layer, initialized to zero.

    Args:
        in_dim (int): The number of input dimensions (channels).
        num_units (int): The number of output units (channels).
        init_scale (float): The initial scale for weight initialization.
            Default is 0.1.

    Returns:
        torch.Tensor: The output tensor after applying the NIN operation.

    Examples:
        >>> nin_layer = NIN(in_dim=64, num_units=128)
        >>> input_tensor = torch.randn(32, 64, 8, 8)  # (batch_size, channels, height, width)
        >>> output_tensor = nin_layer(input_tensor)
        >>> print(output_tensor.shape)  # (32, 128, 8, 8)

    Note:
        The input tensor is permuted before the computation to align with the
        expected dimensions for matrix multiplication.
    """

    def __init__(self, in_dim, num_units, init_scale=0.1):
        super().__init__()
        self.W = nn.Parameter(
            default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True
        )
        self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

    def forward(self, x):
        """
                The NIN class implements a Network in Network (NIN) layer for deep learning.

        This class is a custom PyTorch module that applies a linear transformation
        to the input tensor using learnable parameters. The input tensor is
        expected to be in the shape (batch_size, channels, height, width) and
        is permuted to (batch_size, height, width, channels) before the
        linear transformation is applied. The output is then permuted back to
        the original shape.

        Attributes:
            W (torch.nn.Parameter): Learnable weight parameter of the layer.
            b (torch.nn.Parameter): Learnable bias parameter of the layer.

        Args:
            in_dim (int): The number of input channels.
            num_units (int): The number of output channels (units).
            init_scale (float): Scale for weight initialization. Default is 0.1.

        Returns:
            Tensor: The output tensor after applying the linear transformation.

        Examples:
            >>> import torch
            >>> model = NIN(in_dim=3, num_units=10)
            >>> input_tensor = torch.randn(4, 3, 32, 32)  # Batch of 4 images
            >>> output_tensor = model(input_tensor)
            >>> output_tensor.shape
            torch.Size([4, 10, 32, 32])  # Output shape after NIN
        """
        x = x.permute(0, 2, 3, 1)
        y = contract_inner(x, self.W) + self.b
        return y.permute(0, 3, 1, 2)


class AttnBlock(nn.Module):
    """
    Channel-wise self-attention block.

    This module implements a channel-wise self-attention mechanism using
    normalization and learned linear transformations. It applies
    Group Normalization followed by multiple NIN (Network in Network)
    layers to compute the attention weights and the output.

    Attributes:
        GroupNorm_0 (nn.GroupNorm): Group normalization layer for input.
        NIN_0 (NIN): First NIN layer for query generation.
        NIN_1 (NIN): Second NIN layer for key generation.
        NIN_2 (NIN): Third NIN layer for value generation.
        NIN_3 (NIN): Fourth NIN layer for output transformation.

    Args:
        channels (int): Number of input channels.

    Returns:
        torch.Tensor: The output tensor after applying self-attention.

    Examples:
        >>> attn_block = AttnBlock(channels=64)
        >>> input_tensor = torch.randn(1, 64, 32, 32)  # Batch size 1, 64 channels
        >>> output_tensor = attn_block(input_tensor)
        >>> output_tensor.shape
        torch.Size([1, 64, 32, 32])  # Output has the same shape as input

    Note:
        This block assumes that the input tensor has the shape
        (B, C, H, W) where B is the batch size, C is the number of
        channels, and H and W are the height and width of the feature map.
    """

    def __init__(self, channels):
        super().__init__()
        self.GroupNorm_0 = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)
        self.NIN_0 = NIN(channels, channels)
        self.NIN_1 = NIN(channels, channels)
        self.NIN_2 = NIN(channels, channels)
        self.NIN_3 = NIN(channels, channels, init_scale=0.0)

    def forward(self, x):
        """
            Channel-wise self-attention block.

        This module implements a channel-wise self-attention mechanism, which
        allows the model to focus on different parts of the input feature maps
        based on their relevance. The block utilizes group normalization and
        learnable linear layers (NIN) for transforming the input features.

        Attributes:
            GroupNorm_0: Group normalization layer.
            NIN_0: First learnable linear layer for query transformation.
            NIN_1: Second learnable linear layer for key transformation.
            NIN_2: Third learnable linear layer for value transformation.
            NIN_3: Fourth learnable linear layer for output transformation.

        Args:
            channels (int): Number of input channels.

        Returns:
            Tensor: The output tensor after applying the attention mechanism.

        Examples:
            >>> attn_block = AttnBlock(channels=64)
            >>> input_tensor = torch.randn(8, 64, 32, 32)  # (batch_size, channels, height, width)
            >>> output_tensor = attn_block(input_tensor)
            >>> output_tensor.shape
            torch.Size([8, 64, 32, 32])

        Note:
            The attention weights are computed using the scaled dot-product
            attention mechanism, followed by a softmax normalization.

        Raises:
            ValueError: If the input tensor does not have the expected shape.

        Todo:
            - Consider adding dropout after the attention weights are applied.
        """
        B, C, H, W = x.shape
        h = self.GroupNorm_0(x)
        q = self.NIN_0(h)
        k = self.NIN_1(h)
        v = self.NIN_2(h)

        w = torch.einsum("bchw,bcij->bhwij", q, k) * (int(C) ** (-0.5))
        w = torch.reshape(w, (B, H, W, H * W))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B, H, W, H, W))
        h = torch.einsum("bhwij,bcij->bchw", w, v)
        h = self.NIN_3(h)
        return x + h


class Upsample(nn.Module):
    """
    Upsampling layer with optional convolution.

    This class implements an upsampling operation using nearest neighbor
    interpolation, followed by an optional 3x3 convolution. It is designed to
    double the height and width of the input tensor.

    Attributes:
        Conv_0 (nn.Module, optional): A 3x3 convolution layer applied to the
            upsampled output if `with_conv` is set to True.
        with_conv (bool): Indicates whether to apply a convolution after
            upsampling.

    Args:
        channels (int): The number of input and output channels for the
            convolution layer.
        with_conv (bool): If True, a convolution is applied after the
            upsampling. Defaults to False.

    Returns:
        torch.Tensor: The upsampled tensor, optionally processed by a
        convolution layer.

    Examples:
        >>> upsample_layer = Upsample(channels=64, with_conv=True)
        >>> input_tensor = torch.randn(1, 64, 32, 32)  # Batch of 1, 64 channels
        >>> output_tensor = upsample_layer(input_tensor)
        >>> print(output_tensor.shape)  # Output shape: (1, 64, 64, 64)

    Note:
        The input tensor is expected to have a shape of (B, C, H, W), where
        B is the batch size, C is the number of channels, H is the height,
        and W is the width.
    """

    def __init__(self, channels, with_conv=False):
        super().__init__()
        if with_conv:
            self.Conv_0 = ddpm_conv3x3(channels, channels)
        self.with_conv = with_conv

    def forward(self, x):
        """
            Upsample block for increasing the spatial dimensions of the input.

        This module performs upsampling of the input tensor, optionally followed
        by a 3x3 convolution. It uses nearest neighbor interpolation for upsampling
        and can include a convolutional layer to refine the output.

        Attributes:
            Conv_0 (nn.Conv2d): Optional convolutional layer applied after upsampling.
            with_conv (bool): Flag indicating whether to apply a convolution after
                upsampling.

        Args:
            channels (int): Number of input and output channels for the convolution.
            with_conv (bool): If True, includes a convolutional layer after the
                upsampling.

        Returns:
            Tensor: The upsampled (and optionally convolved) output tensor.

        Examples:
            >>> upsample_layer = Upsample(channels=64, with_conv=True)
            >>> input_tensor = torch.randn(1, 64, 32, 32)  # Batch of 1, 64 channels
            >>> output_tensor = upsample_layer(input_tensor)
            >>> print(output_tensor.shape)  # Output shape should be [1, 64, 64, 64]
        """
        B, C, H, W = x.shape
        h = F.interpolate(x, (H * 2, W * 2), mode="nearest")
        if self.with_conv:
            h = self.Conv_0(h)
        return h


class Downsample(nn.Module):
    """
    Downsample the input tensor using average pooling or convolution.

    This class implements a downsampling layer that reduces the spatial
    dimensions of the input tensor. It can perform downsampling using
    average pooling or a convolutional layer, depending on the
    `with_conv` parameter. When using a convolutional layer, 'SAME'
    padding is applied to ensure the output dimensions are halved.

    Attributes:
        Conv_0 (nn.Module): A 3x3 convolutional layer used for downsampling
            if `with_conv` is set to True.
        with_conv (bool): Indicates whether to use convolution for downsampling
            or average pooling.

    Args:
        channels (int): Number of input channels.
        with_conv (bool): If True, uses convolution for downsampling.
            Defaults to False.

    Returns:
        torch.Tensor: The downsampled tensor.

    Examples:
        >>> downsample_layer = Downsample(channels=64, with_conv=True)
        >>> input_tensor = torch.randn(1, 64, 128, 128)  # (B, C, H, W)
        >>> output_tensor = downsample_layer(input_tensor)
        >>> print(output_tensor.shape)  # Should be (1, 64, 64, 64)

    Note:
        If `with_conv` is set to True, the input tensor will be padded
        before applying the convolution. This ensures that the output size
        is consistent with the downsampling operation.
    """

    def __init__(self, channels, with_conv=False):
        super().__init__()
        if with_conv:
            self.Conv_0 = ddpm_conv3x3(channels, channels, stride=2, padding=0)
        self.with_conv = with_conv

    def forward(self, x):
        """
            Downsample the input tensor using average pooling or convolution.

        This module downsamples the input tensor either by using average pooling
        or by applying a 3x3 convolution with a stride of 2. The option to
        perform convolution can be enabled during initialization.

        Attributes:
            with_conv (bool): A flag to determine whether to use convolution for
                              downsampling.

        Args:
            channels (int): The number of input channels.
            with_conv (bool): If True, uses convolution for downsampling.
                              Defaults to False.

        Returns:
            Tensor: The downsampled tensor of shape (B, C, H//2, W//2), where B is
                    the batch size, C is the number of channels, H is the height,
                    and W is the width of the input tensor.

        Examples:
            >>> downsample_layer = Downsample(channels=64, with_conv=True)
            >>> input_tensor = torch.randn(1, 64, 32, 32)
            >>> output_tensor = downsample_layer(input_tensor)
            >>> output_tensor.shape
            torch.Size([1, 64, 16, 16])

        Note:
            When using convolution for downsampling, the input tensor will be
            padded to emulate 'SAME' padding, ensuring that the output tensor
            has the correct spatial dimensions.

        Raises:
            AssertionError: If the shape of the output tensor does not match the
                            expected shape (B, C, H//2, W//2).
        """
        B, C, H, W = x.shape
        # Emulate 'SAME' padding
        if self.with_conv:
            x = F.pad(x, (0, 1, 0, 1))
            x = self.Conv_0(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)

        assert x.shape == (B, C, H // 2, W // 2)
        return x


class ResnetBlockDDPM(nn.Module):
    """
    The ResNet Blocks used in DDPM.

    This class implements a ResNet block specifically designed for
    diffusion models. It utilizes convolutional layers, group normalization,
    and activation functions to process input tensors. The block supports
    optional time embeddings for enhanced feature representation.

    Attributes:
        GroupNorm_0 (nn.GroupNorm): First group normalization layer.
        act (callable): Activation function applied after normalization.
        Conv_0 (nn.Conv2d): First convolutional layer.
        Dense_0 (nn.Linear): Linear layer for time embedding (if provided).
        GroupNorm_1 (nn.GroupNorm): Second group normalization layer.
        Dropout_0 (nn.Dropout): Dropout layer for regularization.
        Conv_1 (nn.Conv2d): Second convolutional layer.
        Conv_2 (nn.Conv2d): Convolutional layer for shortcut connection (if
            needed).
        out_ch (int): Number of output channels.
        in_ch (int): Number of input channels.
        conv_shortcut (bool): Flag to indicate if a convolutional shortcut
            should be used.

    Args:
        act (callable): Activation function to use.
        in_ch (int): Number of input channels.
        out_ch (int, optional): Number of output channels. Defaults to
            `in_ch`.
        temb_dim (int, optional): Dimension of the time embedding. If
            provided, a linear layer will be added to process it.
        conv_shortcut (bool, optional): If True, use a convolutional layer
            for the shortcut connection; otherwise, use a learnable linear
            layer. Defaults to False.
        dropout (float, optional): Dropout probability. Defaults to 0.1.

    Returns:
        Tensor: The output tensor after passing through the ResNet block.

    Examples:
        >>> block = ResnetBlockDDPM(act=nn.ReLU(), in_ch=64, out_ch=128)
        >>> input_tensor = torch.randn(8, 64, 32, 32)  # (batch, channels, height, width)
        >>> output_tensor = block(input_tensor)
        >>> output_tensor.shape
        torch.Size([8, 128, 32, 32])  # Output shape

    Note:
        The output shape will change based on the `out_ch` parameter.
        If `out_ch` is not specified, it defaults to `in_ch`.

    Raises:
        AssertionError: If the input tensor does not have the expected
            number of channels.
    """

    def __init__(
        self, act, in_ch, out_ch=None, temb_dim=None, conv_shortcut=False, dropout=0.1
    ):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        self.GroupNorm_0 = nn.GroupNorm(num_groups=32, num_channels=in_ch, eps=1e-6)
        self.act = act
        self.Conv_0 = ddpm_conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
            nn.init.zeros_(self.Dense_0.bias)

        self.GroupNorm_1 = nn.GroupNorm(num_groups=32, num_channels=out_ch, eps=1e-6)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = ddpm_conv3x3(out_ch, out_ch, init_scale=0.0)
        if in_ch != out_ch:
            if conv_shortcut:
                self.Conv_2 = ddpm_conv3x3(in_ch, out_ch)
            else:
                self.NIN_0 = NIN(in_ch, out_ch)
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.conv_shortcut = conv_shortcut

    def forward(self, x, temb=None):
        """
            The ResNet Blocks used in DDPM.

        This class implements a ResNet block specifically designed for the
        Denoising Diffusion Probabilistic Models (DDPM). It consists of
        convolutional layers, normalization, activation functions, and
        optional temporal embeddings for enhanced feature representation.

        Attributes:
            act (nn.Module): Activation function to be applied.
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            temb_dim (int, optional): Dimension of the time embedding.
            conv_shortcut (bool): Whether to use a convolutional shortcut.
            dropout (float): Dropout rate to apply after the first convolution.
            GroupNorm_0 (nn.GroupNorm): First group normalization layer.
            Conv_0 (nn.Conv2d): First convolution layer.
            Dense_0 (nn.Linear, optional): Dense layer for time embedding.
            GroupNorm_1 (nn.GroupNorm): Second group normalization layer.
            Dropout_0 (nn.Dropout): Dropout layer.
            Conv_1 (nn.Conv2d): Second convolution layer.
            Conv_2 (nn.Conv2d, optional): Convolution layer for shortcut.
            NIN_0 (NIN, optional): Network-in-Network layer for shortcut.

        Args:
            act (nn.Module): Activation function to use (e.g., nn.ReLU).
            in_ch (int): Number of input channels.
            out_ch (int, optional): Number of output channels. Defaults to in_ch.
            temb_dim (int, optional): Dimension of the temporal embedding.
            conv_shortcut (bool, optional): Use convolutional shortcut. Defaults to False.
            dropout (float, optional): Dropout rate. Defaults to 0.1.

        Returns:
            Tensor: The output of the ResNet block.

        Examples:
            >>> block = ResnetBlockDDPM(act=nn.ReLU(), in_ch=64, out_ch=128)
            >>> input_tensor = torch.randn(8, 64, 32, 32)  # (batch_size, channels, height, width)
            >>> output_tensor = block(input_tensor)
            >>> output_tensor.shape
            torch.Size([8, 128, 32, 32])  # Output shape may vary based on the architecture

        Note:
            The block uses a combination of convolutional layers, group normalization,
            and activation functions to allow for residual learning.

        Raises:
            AssertionError: If the input channels do not match the expected number.
        """
        B, C, H, W = x.shape
        assert C == self.in_ch
        out_ch = self.out_ch if self.out_ch else self.in_ch
        h = self.act(self.GroupNorm_0(x))
        h = self.Conv_0(h)
        # Add bias to each feature map conditioned on the time embedding
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        if C != out_ch:
            if self.conv_shortcut:
                x = self.Conv_2(x)
            else:
                x = self.NIN_0(x)
        return x + h
