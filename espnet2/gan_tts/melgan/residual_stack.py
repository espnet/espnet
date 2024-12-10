# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Residual stack module in MelGAN.

This code is modified from https://github.com/kan-bayashi/ParallelWaveGAN.

"""

from typing import Any, Dict

import torch


class ResidualStack(torch.nn.Module):
    """
        Residual stack module in MelGAN.

    This code is modified from https://github.com/kan-bayashi/ParallelWaveGAN.

    Attributes:
        stack (torch.nn.Sequential): A sequential container for the residual stack.
        skip_layer (torch.nn.Conv1d): A convolutional layer for skip connections.

    Args:
        kernel_size (int): Kernel size of dilation convolution layer.
        channels (int): Number of channels of convolution layers.
        dilation (int): Dilation factor.
        bias (bool): Whether to add bias parameter in convolution layers.
        nonlinear_activation (str): Activation function module name.
        nonlinear_activation_params (Dict[str, Any]): Hyperparameters for
            activation function.
        pad (str): Padding function module name before dilated convolution layer.
        pad_params (Dict[str, Any]): Hyperparameters for padding function.

    Returns:
        None

    Examples:
        >>> residual_stack = ResidualStack(kernel_size=3, channels=32)
        >>> input_tensor = torch.randn(1, 32, 100)  # (B, channels, T)
        >>> output_tensor = residual_stack(input_tensor)
        >>> print(output_tensor.shape)  # Should be (1, 32, 100)

    Raises:
        AssertionError: If the kernel size is an even number.

    Note:
        The residual stack combines convolutional layers with skip connections to
        improve the learning capability of the model.
    """

    def __init__(
        self,
        kernel_size: int = 3,
        channels: int = 32,
        dilation: int = 1,
        bias: bool = True,
        nonlinear_activation: str = "LeakyReLU",
        nonlinear_activation_params: Dict[str, Any] = {"negative_slope": 0.2},
        pad: str = "ReflectionPad1d",
        pad_params: Dict[str, Any] = {},
    ):
        """Initialize ResidualStack module.

        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            channels (int): Number of channels of convolution layers.
            dilation (int): Dilation factor.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (Dict[str, Any]): Hyperparameters for
                activation function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (Dict[str, Any]): Hyperparameters for padding function.

        """
        super().__init__()

        # defile residual stack part
        assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
        self.stack = torch.nn.Sequential(
            getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            getattr(torch.nn, pad)((kernel_size - 1) // 2 * dilation, **pad_params),
            torch.nn.Conv1d(
                channels, channels, kernel_size, dilation=dilation, bias=bias
            ),
            getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            torch.nn.Conv1d(channels, channels, 1, bias=bias),
        )

        # defile extra layer for skip connection
        self.skip_layer = torch.nn.Conv1d(channels, channels, 1, bias=bias)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """
                Residual stack module introduced in MelGAN.

        This code is modified from https://github.com/kan-bayashi/ParallelWaveGAN.

        Attributes:
            stack (torch.nn.Sequential): Sequential container for the residual stack.
            skip_layer (torch.nn.Conv1d): Convolution layer for the skip connection.

        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            channels (int): Number of channels of convolution layers.
            dilation (int): Dilation factor.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (Dict[str, Any]): Hyperparameters for
                activation function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (Dict[str, Any]): Hyperparameters for padding function.

        Examples:
            >>> residual_stack = ResidualStack(kernel_size=3, channels=32)
            >>> input_tensor = torch.randn(1, 32, 100)  # Batch size of 1, 32 channels, 100 time steps
            >>> output_tensor = residual_stack(input_tensor)
            >>> output_tensor.shape
            torch.Size([1, 32, 100])
        """
        return self.stack(c) + self.skip_layer(c)
