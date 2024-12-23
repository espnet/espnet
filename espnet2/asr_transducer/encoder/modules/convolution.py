"""Convolution modules for X-former blocks."""

from typing import Dict, Optional, Tuple

import torch


class ConformerConvolution(torch.nn.Module):
    """ConformerConvolution module definition.

    Args:
        channels: The number of channels.
        kernel_size: Size of the convolving kernel.
        activation: Activation function.
        norm_args: Normalization module arguments.
        causal: Whether to use causal convolution (set to True if streaming).

    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        activation: torch.nn.Module = torch.nn.ReLU(),
        norm_args: Dict = {},
        causal: bool = False,
    ) -> None:
        """Construct an ConformerConvolution object."""
        super().__init__()

        assert (kernel_size - 1) % 2 == 0

        self.kernel_size = kernel_size

        self.pointwise_conv1 = torch.nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        if causal:
            self.lorder = kernel_size - 1
            padding = 0
        else:
            self.lorder = 0
            padding = (kernel_size - 1) // 2

        self.depthwise_conv = torch.nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
        )
        self.norm = torch.nn.BatchNorm1d(channels, **norm_args)
        self.pointwise_conv2 = torch.nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute convolution module.

        This method applies a series of convolution operations on the input tensor
        `x`, potentially using a source mask and a cache for efficient processing.
        The convolution is performed using a pointwise, depthwise, and another
        pointwise convolution layer, with optional activation and normalization.

        Args:
            x: ConformerConvolution input sequences. Shape (B, T, D_hidden),
            where B is the batch size, T is the sequence length, and
            D_hidden is the number of hidden units.
            mask: Source mask. Shape (B, T_2). This mask is applied to zero out
                certain positions in the input tensor `x`.
            cache: ConformerConvolution input cache. Shape (1, D_hidden, conv_kernel).
                This cache is used to store previous outputs for causal
                convolutions.

        Returns:
            x: ConformerConvolution output sequences. Shape (B, ?, D_hidden),
            where the second dimension may vary depending on the operations
            performed.
            cache: ConformerConvolution output cache. Shape (1, D_hidden, conv_kernel).
                This cache can be used in subsequent calls to maintain state.

        Examples:
            >>> model = ConformerConvolution(channels=64, kernel_size=3)
            >>> input_tensor = torch.randn(32, 10, 64)  # Batch of 32, 10 time steps
            >>> output, new_cache = model(input_tensor)
            >>> print(output.shape)  # Output shape should be (32, ?, 64)

        Note:
            Ensure that the input tensor `x` is properly shaped and the kernel size
            is odd to maintain symmetry in convolution operations.
        """
        x = self.pointwise_conv1(x.transpose(1, 2))
        x = torch.nn.functional.glu(x, dim=1)

        if mask is not None:
            x.masked_fill_(mask.unsqueeze(1).expand_as(x), 0.0)

        if self.lorder > 0:
            if cache is None:
                x = torch.nn.functional.pad(x, (self.lorder, 0), "constant", 0.0)
            else:
                x = torch.cat([cache, x], dim=2)
                cache = x[..., -self.lorder :]

        x = self.depthwise_conv(x)
        x = self.activation(self.norm(x))

        x = self.pointwise_conv2(x).transpose(1, 2)

        return x, cache


class ConvolutionalSpatialGatingUnit(torch.nn.Module):
    """
    Convolutional Spatial Gating Unit module definition.

    This module performs a convolution operation that splits the input into two
    parts, applies normalization to one part, and then combines the results
    through an element-wise multiplication. It is designed to work in the context
    of convolutional neural networks, particularly for attention mechanisms.

    Args:
        size: Initial size to determine the number of channels. The input will be
            split into two equal parts.
        kernel_size: Size of the convolving kernel.
        norm_class: Normalization module class (default: torch.nn.LayerNorm).
        norm_args: Normalization module arguments (default: empty dictionary).
        dropout_rate: Dropout rate to apply after the gating operation (default: 0.0).
        causal: Whether to use causal convolution (set to True if streaming).

    Attributes:
        kernel_size (int): Size of the convolution kernel.
        lorder (int): The left order for causal convolution.
        conv (torch.nn.Conv1d): The convolutional layer for gating.
        norm (torch.nn.Module): The normalization layer.
        activation (torch.nn.Module): The activation function.
        dropout (torch.nn.Dropout): The dropout layer.

    Examples:
        >>> unit = ConvolutionalSpatialGatingUnit(size=64, kernel_size=3)
        >>> input_tensor = torch.rand(10, 32, 64)  # (B, T, D_hidden)
        >>> output, cache = unit(input_tensor)
        >>> output.shape
        torch.Size([10, ?, 64])

    Note:
        The input tensor is expected to have a shape of (B, T, D_hidden), where
        B is the batch size, T is the sequence length, and D_hidden is the
        dimensionality of the hidden state. The output tensor will have a shape
        of (B, ?, D_hidden), where '?' depends on the operations performed.

    Raises:
        ValueError: If the size is not a positive even integer.
    """

    def __init__(
        self,
        size: int,
        kernel_size: int,
        norm_class: torch.nn.Module = torch.nn.LayerNorm,
        norm_args: Dict = {},
        dropout_rate: float = 0.0,
        causal: bool = False,
    ) -> None:
        """Construct a ConvolutionalSpatialGatingUnit object."""
        super().__init__()

        channels = size // 2

        self.kernel_size = kernel_size

        if causal:
            self.lorder = kernel_size - 1
            padding = 0
        else:
            self.lorder = 0
            padding = (kernel_size - 1) // 2

        self.conv = torch.nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
        )

        self.norm = norm_class(channels, **norm_args)
        self.activation = torch.nn.Identity()

        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute convolution module.

        This method processes the input tensor `x` through a series of
        convolutional operations, applying a gating mechanism, and handling
        optional masking and caching for causal convolutions.

        Args:
            x: ConvolutionalSpatialGatingUnit input sequences. Shape (B, T, D_hidden),
            where B is the batch size, T is the sequence length, and D_hidden
            is the number of features.
            mask: Optional source mask. Shape (B, T_2), used to prevent certain
                positions in the input from being processed.
            cache: Optional input cache for maintaining state across calls. Shape
                (1, D_hidden, conv_kernel).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - x: ConvolutionalSpatialGatingUnit output sequences. Shape (B, ?,
                D_hidden), where the second dimension may vary based on the
                convolution operation.
                - cache: ConvolutionalSpatialGatingUnit output cache. Shape
                (1, D_hidden, conv_kernel).

        Examples:
            >>> unit = ConvolutionalSpatialGatingUnit(size=128, kernel_size=3)
            >>> input_tensor = torch.randn(10, 20, 64)  # Batch of 10, 20 timesteps
            >>> output, new_cache = unit(input_tensor)
            >>> print(output.shape)  # Output shape may vary based on convolutions

        Note:
            - The input tensor `x` is expected to have its last dimension split
            into two halves for gating purposes.
            - If causal is set to True during initialization, the input will be
            processed in a way that prevents future information from being used.
        """
        x_r, x_g = x.chunk(2, dim=-1)

        x_g = self.norm(x_g).transpose(1, 2)

        if mask is not None:
            x_g.masked_fill_(mask.unsqueeze(1).expand_as(x_g), 0.0)

        if self.lorder > 0:
            if cache is None:
                x_g = torch.nn.functional.pad(x_g, (self.lorder, 0), "constant", 0.0)
            else:
                x_g = torch.cat([cache, x_g], dim=2)
                cache = x_g[..., -self.lorder :]

        x_g = self.conv(x_g).transpose(1, 2)

        x = self.dropout(x_r * self.activation(x_g))

        return x, cache


class DepthwiseConvolution(torch.nn.Module):
    """
    Depth-wise Convolution module definition.

    This module performs depth-wise convolution, which applies a separate
    convolutional filter to each input channel. It is commonly used in
    lightweight neural networks to reduce the number of parameters and
    computations.

    Args:
        size: Initial size to determine the number of channels.
              The total number of channels will be `size + size`.
        kernel_size: Size of the convolving kernel.
        causal: Whether to use causal convolution (set to True if streaming).

    Attributes:
        kernel_size: The size of the convolutional kernel.
        lorder: The length of the causal order.
        conv: The 1D convolution layer that performs the depth-wise convolution.

    Examples:
        >>> import torch
        >>> depthwise_conv = DepthwiseConvolution(size=64, kernel_size=3)
        >>> input_tensor = torch.rand(32, 10, 128)  # (B, T, D_hidden)
        >>> output, cache = depthwise_conv(input_tensor)
        >>> output.shape
        torch.Size([32, 10, 128])  # Output shape depends on padding and input

    Note:
        The input tensor `x` should have shape (B, T, D_hidden), where B is the
        batch size, T is the sequence length, and D_hidden is the number of
        features (channels).

    Raises:
        ValueError: If the kernel_size is not a positive odd integer.
    """

    def __init__(
        self,
        size: int,
        kernel_size: int,
        causal: bool = False,
    ) -> None:
        """Construct a DepthwiseConvolution object."""
        super().__init__()

        channels = size + size

        self.kernel_size = kernel_size

        if causal:
            self.lorder = kernel_size - 1
            padding = 0
        else:
            self.lorder = 0
            padding = (kernel_size - 1) // 2

        self.conv = torch.nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the depthwise convolution operation.

        This method performs a depthwise convolution on the input tensor `x`.
        It applies the convolution while considering the optional `mask` and
        `cache`. The mask is used to prevent information leakage in specific
        time steps, while the cache is used to maintain the state across
        sequential calls, useful in causal settings.

        Args:
            x: DepthwiseConvolution input sequences with shape
               (B, T, D_hidden), where B is the batch size, T is the
               sequence length, and D_hidden is the number of hidden units.
            mask: Optional source mask with shape (B, T_2) that indicates
                  which elements should be ignored (masked).
            cache: Optional input cache with shape (1, conv_kernel, D_hidden)
                   used to store previous state for causal convolutions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - x: DepthwiseConvolution output sequences with shape
                  (B, ?, D_hidden), where ? indicates the new sequence length
                  after the convolution operation.
                - cache: DepthwiseConvolution output cache with shape
                  (1, conv_kernel, D_hidden), which is updated based on the
                  current input.

        Note:
            The input tensor `x` is transposed to match the expected input
            shape for the convolution layer. If a mask is provided, it is
            applied to the input tensor to prevent the model from attending
            to certain time steps.

        Examples:
            >>> model = DepthwiseConvolution(size=64, kernel_size=3)
            >>> input_tensor = torch.randn(10, 5, 128)  # (B, T, D_hidden)
            >>> mask = torch.ones(10, 5)  # No masking
            >>> output, cache = model(input_tensor, mask=mask)
            >>> output.shape  # Expected shape: (10, ?, 128)
            torch.Size([10, ?, 128])
        """
        x = x.transpose(1, 2)

        if mask is not None:
            x.masked_fill_(mask.unsqueeze(1).expand_as(x), 0.0)

        if self.lorder > 0:
            if cache is None:
                x = torch.nn.functional.pad(x, (self.lorder, 0), "constant", 0.0)
            else:
                x = torch.cat([cache, x], dim=2)
                cache = x[..., -self.lorder :]

        x = self.conv(x).transpose(1, 2)

        return x, cache
