# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Flow-related modules of VITS.

This code is based on the official implementation:
- https://github.com/jaywalnut310/vits

"""

import math

import torch

from espnet2.tts.vits.transform import piecewise_rational_quadratic_transform
from espnet2.tts.vits.wavenet import WaveNet


class Flip(torch.nn.Module):
    """Flip module."""

    def forward(self, x, *args, inverse=False, **kwargs):
        """Calculate forward propagation."""
        x = torch.flip(x, [1])
        if not inverse:
            logdet = x.new_zeros(x.size(0))
            return x, logdet
        else:
            return x


class LogFlow(torch.nn.Module):
    """Log flow module."""

    def forward(self, x, x_mask, inverse=False, eps=1e-5, **kwargs):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, channels, T).
            x_lengths (Tensor): Length tensor (B,).
            inverse (bool): Whether to inverse the flow.
            eps (float): Epsilon for log.

        Returns:
            Tensor: Tensor (B, channels, T).
            Tensor: Determinant tensor (B,).

        """
        if not inverse:
            y = torch.log(torch.clamp_min(x, eps)) * x_mask
            logdet = torch.sum(-y, [1, 2])
            return y, logdet
        else:
            x = torch.exp(x) * x_mask
            return x


class ElementwiseAffineFlow(torch.nn.Module):
    """Element affine flow module."""

    def __init__(self, channels):
        """Initialize ElementwiseAffineFlow module."""
        super().__init__()
        self.channels = channels
        self.register_parameter("m", torch.zeros(channels, 1))
        self.register_parameter("logs", torch.zeros(channels, 1))

    def forward(self, x, x_mask, inverse=False, **kwargs):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, channels, T).
            x_lengths (Tensor): Length tensor (B,).
            inverse (bool): Whether to inverse the flow.

        Returns:
            Tensor: Tensor (B, channels, T).
            Tensor: Determinant tensor (B,).

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
    """Transpose module."""

    def __init__(self, dim1, dim2):
        """Initialize Transpose module."""
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        """Transpose."""
        return x.transpose(self.dim1, self.dim2)


class DilatedDepthSeparableConv(torch.nn.Module):
    """Dilated depth-separable conv module."""

    def __init__(
        self,
        channels,
        kernel_size,
        layers,
        dropout_rate=0.0,
        eps=1e-5,
    ):
        """Initialize DilatedDepthSeparableConv module.

        Args:
            channels: Number of channels.
            kernel_size: Kernel size.
            layers: Number of layers.
            dropout_rate: Dropout rate.
            eps: Epsilon for layer norm.

        """
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for i in range(layers):
            dilation = kernel_size ** i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_sep += [
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

    def forward(self, x, x_mask, g=None):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, in_channels, T).
            x_lengths (Tensor): Length tensor (B,).
            g (Optional[Tensor]): Global conditioning tensor (B, global_channels, 1).

        Returns:
            Tensor: Output tensor (B, channels, T).

        """
        if g is not None:
            x = x + g
        for f in range(self.convs):
            y = f(x * x_mask)
            x = x + y
        return x * x_mask


class ConvFlow(torch.nn.Module):
    """Convolutional flow module."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        kernel_size,
        layers,
        bins=10,
        tail_bound=5.0,
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
        self.output_conv = torch.nn.Conv1d(
            hidden_channels,
            self.half_channels * (bins * 3 - 1),
            1,
        )
        self.output_conv.weight.data.zero_()
        self.output_conv.bias.data.zero_()

    def forward(self, x, x_mask, g=None, inverse=False):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, channels, T).
            x_lengths (Tensor): Length tensor (B,).
            g (Optional[Tensor]): Global conditioning tensor (B, channels, 1)
            inverse (bool): Whether to inverse the flow.

        Returns:
            Tensor: Tensor (B, channels, T).
            Tensor: Determinant tensor (B,).

        """
        xa, xb = x.split(x.size(1) // 2, 1)
        h = self.input_conv(xa)
        h = self.dds_conv(h, x_mask, g=g)
        h = self.output_conv(h) * x_mask  # (B, half_channels * (bins * 3 - 1), T)

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


class ResidualAffineCouplingLayer(torch.nn.Module):
    """Residual affine coupling layer."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        kernel_size=5,
        base_dilation=1,
        layers=5,
        stacks=1,
        global_channels=-1,
        dropout_rate=0.0,
        use_weight_norm=True,
        bias=True,
        use_only_mean=False,
    ):
        """Initialzie ResidualAffineCouplingLayer module.

        Args:
            in_channels (int): Number of input channels.
            hidden_channels: Number of hidden channels.
            kernel_size: Kernel size for WaveNet.
            base_dilation: Base dilation factor for WaveNet.
            layers: Number of layers of WaveNet.
            stacks: Number of stacks of WaveNet.
            global_channels: Number of global channels.
            dropout_rate: Dropout rate.
            use_weight_norm: Whether to use weight normalization in WaveNet.
            bias: Whether to use bias paramters in WaveNet.
            use_only_mean: Whether to estimate only VAE mean.

        """
        assert in_channels % 2 == 0, "in_channels should be divisible by 2"
        super().__init__()
        self.half_channels = in_channels // 2
        self.use_only_mean = use_only_mean

        # define modules
        self.input_conv = torch.nn.Conv1d(
            self.half_channels,
            hidden_channels,
            1,
        )
        self.encoder = WaveNet(
            in_channels=-1,
            out_channels=-1,
            kernel_size=kernel_size,
            layers=layers,
            stacks=stacks,
            base_dilation=base_dilation,
            residual_channels=hidden_channels,
            aux_channels=-1,
            gate_channels=hidden_channels * 2,
            skip_channels=-1,
            global_channels=global_channels,
            dropout_rate=dropout_rate,
            bias=bias,
            use_weight_norm=use_weight_norm,
            use_first_conv=False,
            use_last_conv=False,
        )
        if use_only_mean:
            self.output_conv = torch.nn.Conv1d(
                hidden_channels,
                self.half_channels,
                1,
            )
        else:
            self.output_conv = torch.nn.Conv1d(
                hidden_channels,
                self.half_channels * 2,
                1,
            )

        # initialize output conv
        self.output_conv.weight.data.zero_()
        self.output_conv.bias.data.zero_()

    def forward(self, x, x_mask, g=None, inverse=False):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, in_channels, T).
            x_lengths (Tensor): Length tensor (B,).
            g (Optional[Tensor]): Global conditioning tensor (B, global_channels, 1).
            inverse (bool): Whether to inverse the flow.

        Returns:
            Tensor: Tensor (B, in_channels, T).
            Tensor: Determinant tensor (B,).

        """
        xa, xb = x.split(x.size(1) // 2, dim=1)
        h = self.input_conv(xa) * x_mask
        h = self.encoder(h, x_mask, g=g)
        stats = self.output_conv(h) * x_mask
        if not self.use_only_mean:
            m, logs = stats.split(stats.size(1) // 2, dim=1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not inverse:
            xb = m + xb * torch.exp(logs) * x_mask
            x = torch.cat([xa, xb], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            xb = (xb - m) * torch.exp(-logs) * x_mask
            x = torch.cat([xa, xb], 1)
            return x


class ResidualAffineCouplingBlock(torch.nn.Module):
    """Residual affine coupling block."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        flows,
        kernel_size=5,
        base_dilation=1,
        layers=5,
        global_channels=-1,
        dropout_rate=0.0,
        use_weight_norm=True,
        bias=True,
        use_only_mean=False,
    ):
        """Initilize ResidualAffineCouplingBlock module.

        Args:
            in_channels (int): Number of input channels.
            hidden_channels: Number of hidden channels.
            flows (int): Number of flows.
            kernel_size: Kernel size for WaveNet.
            base_dilation: Base dilation factor for WaveNet.
            layers: Number of layers of WaveNet.
            stacks: Number of stacks of WaveNet.
            global_channels: Number of global channels.
            dropout_rate: Dropout rate.
            use_weight_norm: Whether to use weight normalization in WaveNet.
            bias: Whether to use bias paramters in WaveNet.
            use_only_mean: Whether to estimate only VAE mean.

        """
        super().__init__()

        self.flows = torch.nn.ModuleList()
        for i in range(flows):
            self.flows += [
                ResidualAffineCouplingLayer(
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    kernel_size=kernel_size,
                    base_dilation=base_dilation,
                    layers=layers,
                    stacks=1,
                    global_channels=global_channels,
                    dropout_rate=dropout_rate,
                    use_weight_norm=use_weight_norm,
                    bias=bias,
                    use_only_mean=use_only_mean,
                )
            ]
            self.flows += [Flip()]

    def forward(self, x, x_mask, g=None, inverse=False):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, in_channels, T).
            x_lengths (Tensor): Length tensor (B,).
            g (Optional[Tensor]): Global conditioning tensor (B, global_channels, 1).
            inverse (bool): Whether to inverse the flow.

        Returns:
            Tensor: Processed tensor (B, in_channels, T).

        """
        if not inverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, inverse=inverse)
        else:
            for flow in inversed(self.flows):
                x = flow(x, x_mask, g=g, inverse=inverse)
        return x
