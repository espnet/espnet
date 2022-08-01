"""Convolution for X-former block."""

from typing import Optional, Tuple

import torch


class ConformerConvolution(torch.nn.Module):
    """ConformerConvolution module definition.

    Args:
        channels: The number of channels of conv layers.
        kernel_size: Kernel size of conv layers.
        activation: Type of activation function.
        norm_class: Normalization class.
        norm_eps: Initial epsilon value for the normalization class.
        causal: Whether to use causal convolution (set to True if streaming).

    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        activation: torch.nn.Module = torch.nn.ReLU(),
        norm_class: torch.nn.Module = torch.nn.BatchNorm1d,
        norm_eps: float = 1e-05,
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
        self.norm = norm_class(channels, eps=norm_eps)
        self.pointwise_conv2 = torch.nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
        right_context: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute convolution module.

        Args:
            x: ConformerConvolution input sequences. (B, T, D_hidden)
            cache: ConformerConvolution input cache. (1, conv_kernel, D_hidden)
            right_context: Number of frames in right context.

        Returns:
            x: ConformerConvolution output sequences. (B, T, D_hidden)
            cache: ConformerConvolution output cache. (1, conv_kernel, D_hidden)

        """
        x = self.pointwise_conv1(x.transpose(1, 2))
        x = torch.nn.functional.glu(x, dim=1)

        if self.lorder > 0:
            if cache is None:
                x = torch.nn.functional.pad(x, (self.lorder, 0), "constant", 0.0)
            else:
                x = torch.cat([cache, x], dim=2)

                if right_context > 0:
                    cache = x[:, :, -(self.lorder + right_context) : -right_context]
                else:
                    cache = x[:, :, -self.lorder :]

        x = self.depthwise_conv(x)
        x = self.activation(self.norm(x))

        x = self.pointwise_conv2(x).transpose(1, 2)

        return x, cache
