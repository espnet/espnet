# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Flow-related modules of VITS.

This code is based on the official implementation:
- https://github.com/jaywalnut310/vits

"""

import torch

from espnet2.tts.vits.wavenet import WaveNet


class Flip(torch.nn.Module):
    """Flip module."""

    def forward(self, x, *args, reverse=False, **kwargs):
        """Calculate forward propagation."""
        x = torch.flip(x, [1])
        if not reverse:
            logdet = x.new_zeros(x.size(0))
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

    def forward(self, x, x_mask, g=None, reverse=False):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, in_channels, T).
            x_lengths (Tensor): Length tensor (B,).
            g (Optional[Tensor]): Global conditioning tensor (B, global_channels, 1).
            reverse (bool): Whether to reverse the flow.

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

        if not reverse:
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

    def forward(self, x, x_mask, g=None, reverse=False):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, in_channels, T).
            x_lengths (Tensor): Length tensor (B,).
            g (Optional[Tensor]): Global conditioning tensor (B, global_channels, 1).
            reverse (bool): Whether to reverse the flow.

        Returns:
            Tensor: Processed tensor (B, in_channels, T).

        """
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x
