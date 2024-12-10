# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Basic Flow modules used in VITS.

This code is based on https://github.com/jaywalnut310/vits.

"""

import math
from typing import Optional, Tuple, Union

import torch

from espnet2.gan_tts.vits.transform import piecewise_rational_quadratic_transform


class FlipFlow(torch.nn.Module):
    """
        FlipFlow is a flip flow module that applies a simple flipping operation on
    the input tensor during forward propagation. This module is part of the
    VITS architecture used for generative adversarial networks in text-to-speech
    tasks.

    Attributes:
        None

    Args:
        x (torch.Tensor): Input tensor of shape (B, channels, T).
        inverse (bool): Whether to perform the inverse operation. Default is False.

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            - If `inverse` is False:
                - Tensor: Flipped tensor of shape (B, channels, T).
                - Tensor: Log-determinant tensor for negative log-likelihood (NLL) of shape (B,).
            - If `inverse` is True:
                - Tensor: The original input tensor after flipping.

    Examples:
        >>> flip_flow = FlipFlow()
        >>> x = torch.randn(4, 3, 5)  # Example input tensor
        >>> flipped, logdet = flip_flow(x)  # Forward operation
        >>> original = flip_flow(x, inverse=True)  # Inverse operation

    Note:
        This module assumes that the input tensor has at least 3 dimensions
        corresponding to the batch size, number of channels, and sequence length.
    """

    def forward(
        self, x: torch.Tensor, *args, inverse: bool = False, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
                Calculate forward propagation.

        This method performs the forward pass of the FlipFlow module, which applies a
        flipping transformation to the input tensor. It can also perform the inverse
        operation based on the `inverse` flag.

        Args:
            x (torch.Tensor): Input tensor of shape (B, channels, T).
            *args: Additional positional arguments (unused).
            inverse (bool, optional): Whether to inverse the flow. Defaults to False.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - If `inverse` is False, returns a tuple containing:
                    - Flipped tensor of shape (B, channels, T).
                    - Log-determinant tensor for negative log-likelihood (NLL) of shape (B,).
                - If `inverse` is True, returns the tensor after applying the inverse
                  transformation.

        Examples:
            >>> flip_flow = FlipFlow()
            >>> input_tensor = torch.randn(5, 3, 10)  # Example input
            >>> output_tensor, logdet = flip_flow(input_tensor)
            >>> inverted_tensor = flip_flow(input_tensor, inverse=True)
        """
        x = torch.flip(x, [1])
        if not inverse:
            logdet = x.new_zeros(x.size(0))
            return x, logdet
        else:
            return x


class LogFlow(torch.nn.Module):
    """
        LogFlow module for calculating forward propagation in flow-based models.

    This class implements a log flow transformation as part of the Variational
    Inference Text-to-Speech (VITS) model. It computes the log of the input tensor
    while applying a mask and calculates the log-determinant necessary for
    negative log-likelihood (NLL) computation.

    Args:
        x (Tensor): Input tensor of shape (B, channels, T).
        x_mask (Tensor): Mask tensor of shape (B, 1, T).
        inverse (bool): Whether to compute the inverse of the flow.
        eps (float): A small value to prevent log(0). Default is 1e-5.

    Returns:
        Union[Tensor, Tuple[Tensor, Tensor]]:
            - If `inverse` is False:
                - Output tensor of shape (B, channels, T).
                - Log-determinant tensor for NLL of shape (B,).
            - If `inverse` is True:
                - Output tensor of shape (B, channels, T).

    Examples:
        >>> log_flow = LogFlow()
        >>> x = torch.tensor([[[0.1, 0.2, 0.3]]])  # Shape: (1, 1, 3)
        >>> x_mask = torch.tensor([[[1, 1, 1]]])  # Shape: (1, 1, 3)
        >>> y, logdet = log_flow(x, x_mask)  # Forward propagation
        >>> x_inv = log_flow(y, x_mask, inverse=True)  # Inverse propagation

    Note:
        The log transformation applied here is element-wise, and the log-determinant
        is computed by summing the negative log of the transformed input tensor
        over the time and channel dimensions.
    """

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        inverse: bool = False,
        eps: float = 1e-5,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, channels, T).
            x_mask (Tensor): Mask tensor (B, 1, T).
            inverse (bool): Whether to inverse the flow.
            eps (float): Epsilon for log.

        Returns:
            Tensor: Output tensor (B, channels, T).
            Tensor: Log-determinant tensor for NLL (B,) if not inverse.

        """
        if not inverse:
            y = torch.log(torch.clamp_min(x, eps)) * x_mask
            logdet = torch.sum(-y, [1, 2])
            return y, logdet
        else:
            x = torch.exp(x) * x_mask
            return x


class ElementwiseAffineFlow(torch.nn.Module):
    """
        Elementwise affine flow module.

    This module applies an elementwise affine transformation to the input tensor.
    It allows for the modeling of complex distributions by learning parameters for
    affine transformations.

    Attributes:
        channels (int): Number of channels in the input tensor.
        m (torch.nn.Parameter): Learnable parameter for the affine transformation.
        logs (torch.nn.Parameter): Learnable parameter for the logarithm of the
            scale in the affine transformation.

    Args:
        channels (int): Number of channels.

    Methods:
        forward(x: torch.Tensor, x_mask: torch.Tensor, inverse: bool = False,
                **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            Calculate forward propagation.

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            If inverse is False, returns the transformed tensor and the log-determinant
            for NLL. If inverse is True, returns the original input tensor.

    Examples:
        >>> flow = ElementwiseAffineFlow(channels=10)
        >>> x = torch.randn(2, 10, 5)  # (B, channels, T)
        >>> x_mask = torch.ones(2, 1, 5)  # (B, 1, T)
        >>> output, logdet = flow(x, x_mask)  # Forward transformation
        >>> original = flow(x, x_mask, inverse=True)  # Inverse transformation

    Note:
        The log-determinant is useful for calculating the negative log-likelihood
        during training of generative models.
    """

    def __init__(self, channels: int):
        """Initialize ElementwiseAffineFlow module.

        Args:
            channels (int): Number of channels.

        """
        super().__init__()
        self.channels = channels
        self.register_parameter("m", torch.nn.Parameter(torch.zeros(channels, 1)))
        self.register_parameter("logs", torch.nn.Parameter(torch.zeros(channels, 1)))

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, inverse: bool = False, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
                Elementwise affine flow module.

        This module implements an elementwise affine flow for transforming input
        tensors. It applies an affine transformation defined by learned parameters
        to the input tensor and optionally computes the log-determinant for use in
        normalizing flows.

        Attributes:
            channels (int): Number of channels for the input tensor.
            m (torch.nn.Parameter): Learnable parameter for the affine transformation.
            logs (torch.nn.Parameter): Learnable parameter for the log scale.

        Args:
            channels (int): Number of channels.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - Output tensor (B, channels, T) after the affine transformation.
                - Log-determinant tensor for NLL (B,) if not inverse.

        Examples:
            >>> flow = ElementwiseAffineFlow(channels=5)
            >>> x = torch.randn(2, 5, 10)  # Batch size of 2, 5 channels, length 10
            >>> x_mask = torch.ones(2, 1, 10)  # Mask for all timesteps
            >>> y, logdet = flow(x, x_mask)  # Forward transformation
            >>> x_reconstructed = flow(y, x_mask, inverse=True)  # Inverse transformation
        """
        if not inverse:
            y = self.m + torch.exp(self.logs) * x
            y = y * x_mask
            logdet = torch.sum(self.logs * x_mask, [1, 2])
            return y, logdet
        else:
            x = (x - self.m) * torch.exp(-self.logs) * x_mask
            return x


class Transpose(torch.nn.Module):
    """
        Transpose module for torch.nn.Sequential().

    This module transposes the specified dimensions of the input tensor.

    Attributes:
        dim1 (int): The first dimension to transpose.
        dim2 (int): The second dimension to transpose.

    Args:
        dim1 (int): The first dimension to be transposed.
        dim2 (int): The second dimension to be transposed.

    Returns:
        Tensor: The transposed tensor.

    Examples:
        >>> transpose = Transpose(1, 2)
        >>> input_tensor = torch.randn(2, 3, 4)  # Shape: (B, C, T)
        >>> output_tensor = transpose(input_tensor)
        >>> print(output_tensor.shape)  # Shape: (B, T, C)
    """

    def __init__(self, dim1: int, dim2: int):
        """Initialize Transpose module."""
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transpose module for torch.nn.Sequential().
        """
        return x.transpose(self.dim1, self.dim2)


class DilatedDepthSeparableConv(torch.nn.Module):
    """
        Dilated depth-separable convolution module.

    This module implements a dilated depth-separable convolution, which consists of
    multiple layers of dilated convolutions followed by normalization and activation
    functions. It is designed to process 1D input tensors commonly used in
    sequence-based tasks.

    Attributes:
        convs (ModuleList): A list of sequential convolutional layers.

    Args:
        channels (int): Number of channels.
        kernel_size (int): Size of the convolution kernel.
        layers (int): Number of convolutional layers to stack.
        dropout_rate (float): Rate of dropout to apply after each layer.
        eps (float): Small constant for numerical stability in layer normalization.

    Examples:
        >>> dsc = DilatedDepthSeparableConv(channels=64, kernel_size=3, layers=2)
        >>> x = torch.randn(10, 64, 100)  # (B, channels, T)
        >>> x_mask = torch.ones(10, 1, 100)  # Mask tensor
        >>> output = dsc(x, x_mask)
        >>> print(output.shape)  # Should be (10, 64, 100)

    Returns:
        Tensor: Output tensor with the same shape as input (B, channels, T).
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        layers: int,
        dropout_rate: float = 0.0,
        eps: float = 1e-5,
    ):
        """Initialize DilatedDepthSeparableConv module.

        Args:
            channels (int): Number of channels.
            kernel_size (int): Kernel size.
            layers (int): Number of layers.
            dropout_rate (float): Dropout rate.
            eps (float): Epsilon for layer norm.

        """
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for i in range(layers):
            dilation = kernel_size**i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        groups=channels,
                        dilation=dilation,
                        padding=padding,
                    ),
                    Transpose(1, 2),
                    torch.nn.LayerNorm(
                        channels,
                        eps=eps,
                        elementwise_affine=True,
                    ),
                    Transpose(1, 2),
                    torch.nn.GELU(),
                    torch.nn.Conv1d(
                        channels,
                        channels,
                        1,
                    ),
                    Transpose(1, 2),
                    torch.nn.LayerNorm(
                        channels,
                        eps=eps,
                        elementwise_affine=True,
                    ),
                    Transpose(1, 2),
                    torch.nn.GELU(),
                    torch.nn.Dropout(dropout_rate),
                )
            ]

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, g: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, in_channels, T).
            x_mask (Tensor): Mask tensor (B, 1, T).
            g (Optional[Tensor]): Global conditioning tensor (B, global_channels, 1).

        Returns:
            Tensor: Output tensor (B, channels, T).

        """
        if g is not None:
            x = x + g
        for f in self.convs:
            y = f(x * x_mask)
            x = x + y
        return x * x_mask


class ConvFlow(torch.nn.Module):
    """
        Convolutional flow module for generative modeling.

    This module is part of a series of flow-based transformations used in the VITS
    model. It implements a convolutional flow that maps input tensors through a series
    of transformations.

    Attributes:
        half_channels (int): Number of input channels divided by two.
        hidden_channels (int): Number of hidden channels.
        bins (int): Number of bins for the transformation.
        tail_bound (float): Tail bound value for the transformation.

    Args:
        in_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden channels.
        kernel_size (int): Kernel size for the convolution.
        layers (int): Number of layers in the convolutional flow.
        bins (int, optional): Number of bins for the transformation (default: 10).
        tail_bound (float, optional): Tail bound value for the transformation
                                       (default: 5.0).

    Returns:
        Union[Tensor, Tuple[Tensor, Tensor]]: Output tensor if not inverse, or a tuple
        containing the output tensor and the log-determinant tensor for NLL if inverse
        is False.

    Examples:
        >>> conv_flow = ConvFlow(in_channels=64, hidden_channels=128, kernel_size=3, layers=4)
        >>> x = torch.randn(32, 64, 100)  # (B, channels, T)
        >>> x_mask = torch.ones(32, 1, 100)  # (B, 1, T)
        >>> output, logdet = conv_flow(x, x_mask)  # Forward propagation
        >>> output_inv = conv_flow(x, x_mask, inverse=True)  # Inverse propagation

    Note:
        This implementation relies on a piecewise rational quadratic transform
        for the final mapping of the input.

    Todo:
        Understand the calculation in the forward method, specifically the
        purpose of the `denom` and how it affects the transformations.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        layers: int,
        bins: int = 10,
        tail_bound: float = 5.0,
    ):
        """Initialize ConvFlow module.

        Args:
            in_channels (int): Number of input channels.
            hidden_channels (int): Number of hidden channels.
            kernel_size (int): Kernel size.
            layers (int): Number of layers.
            bins (int): Number of bins.
            tail_bound (float): Tail bound value.

        """
        super().__init__()
        self.half_channels = in_channels // 2
        self.hidden_channels = hidden_channels
        self.bins = bins
        self.tail_bound = tail_bound

        self.input_conv = torch.nn.Conv1d(
            self.half_channels,
            hidden_channels,
            1,
        )
        self.dds_conv = DilatedDepthSeparableConv(
            hidden_channels,
            kernel_size,
            layers,
            dropout_rate=0.0,
        )
        self.proj = torch.nn.Conv1d(
            hidden_channels,
            self.half_channels * (bins * 3 - 1),
            1,
        )
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        inverse: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
            Convolutional flow module.

        This module implements a convolutional flow for generative modeling. It
        transforms an input tensor using a series of convolutional layers and
        piecewise rational quadratic transformations.

        Args:
            in_channels (int): Number of input channels.
            hidden_channels (int): Number of hidden channels.
            kernel_size (int): Kernel size.
            layers (int): Number of layers.
            bins (int): Number of bins.
            tail_bound (float): Tail bound value.

        Attributes:
            half_channels (int): Half of the input channels.
            hidden_channels (int): Number of hidden channels.
            bins (int): Number of bins.
            tail_bound (float): Tail bound value.
            input_conv (torch.nn.Conv1d): Initial convolution layer.
            dds_conv (DilatedDepthSeparableConv): Dilated depth-separable conv layer.
            proj (torch.nn.Conv1d): Projection layer.

        Args:
            x (Tensor): Input tensor (B, channels, T).
            x_mask (Tensor): Mask tensor (B,).
            g (Optional[Tensor]): Global conditioning tensor (B, channels, 1).
            inverse (bool): Whether to inverse the flow.

        Returns:
            Tensor: Output tensor (B, channels, T).
            Tensor: Log-determinant tensor for NLL (B,) if not inverse.

        Examples:
            >>> conv_flow = ConvFlow(in_channels=16, hidden_channels=32, kernel_size=3,
            ...                       layers=2, bins=10, tail_bound=5.0)
            >>> x = torch.randn(8, 16, 100)  # Batch of 8, 16 channels, 100 time steps
            >>> x_mask = torch.ones(8, 1, 100)  # Mask with all values as 1
            >>> output, logdet = conv_flow(x, x_mask)
            >>> print(output.shape)  # Should be (8, 16, 100)
            >>> print(logdet.shape)  # Should be (8,)

        Note:
            The `piecewise_rational_quadratic_transform` function is used to
            compute the transformation of the second half of the input tensor.
        """
        xa, xb = x.split(x.size(1) // 2, 1)
        h = self.input_conv(xa)
        h = self.dds_conv(h, x_mask, g=g)
        h = self.proj(h) * x_mask  # (B, half_channels * (bins * 3 - 1), T)

        b, c, t = xa.shape
        # (B, half_channels, bins * 3 - 1, T) -> (B, half_channels, T, bins * 3 - 1)
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)

        # TODO(kan-bayashi): Understand this calculation
        denom = math.sqrt(self.hidden_channels)
        unnorm_widths = h[..., : self.bins] / denom
        unnorm_heights = h[..., self.bins : 2 * self.bins] / denom
        unnorm_derivatives = h[..., 2 * self.bins :]
        xb, logdet_abs = piecewise_rational_quadratic_transform(
            xb,
            unnorm_widths,
            unnorm_heights,
            unnorm_derivatives,
            inverse=inverse,
            tails="linear",
            tail_bound=self.tail_bound,
        )
        x = torch.cat([xa, xb], 1) * x_mask
        logdet = torch.sum(logdet_abs * x_mask, [1, 2])
        if not inverse:
            return x, logdet
        else:
            return x
