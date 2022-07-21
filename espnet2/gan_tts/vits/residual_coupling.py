# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Residual affine coupling modules in VITS.

This code is based on https://github.com/jaywalnut310/vits.

"""

from typing import Optional, Tuple, Union

import torch

from espnet2.gan_tts.vits.flow import FlipFlow
from espnet2.gan_tts.wavenet import WaveNet


class ResidualAffineCouplingBlock(torch.nn.Module):
    """Residual affine coupling block module.

    This is a module of residual affine coupling block, which used as "Flow" in
    `Conditional Variational Autoencoder with Adversarial Learning for End-to-End
    Text-to-Speech`_.

    .. _`Conditional Variational Autoencoder with Adversarial Learning for End-to-End
        Text-to-Speech`: https://arxiv.org/abs/2006.04558

    """

    def __init__(
        self,
        in_channels: int = 192,
        hidden_channels: int = 192,
        flows: int = 4,
        kernel_size: int = 5,
        base_dilation: int = 1,
        layers: int = 4,
        global_channels: int = -1,
        dropout_rate: float = 0.0,
        use_weight_norm: bool = True,
        bias: bool = True,
        use_only_mean: bool = True,
    ):
        """Initilize ResidualAffineCouplingBlock module.

        Args:
            in_channels (int): Number of input channels.
            hidden_channels (int): Number of hidden channels.
            flows (int): Number of flows.
            kernel_size (int): Kernel size for WaveNet.
            base_dilation (int): Base dilation factor for WaveNet.
            layers (int): Number of layers of WaveNet.
            stacks (int): Number of stacks of WaveNet.
            global_channels (int): Number of global channels.
            dropout_rate (float): Dropout rate.
            use_weight_norm (bool): Whether to use weight normalization in WaveNet.
            bias (bool): Whether to use bias paramters in WaveNet.
            use_only_mean (bool): Whether to estimate only mean.

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
            self.flows += [FlipFlow()]

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        inverse: bool = False,
    ) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, in_channels, T).
            x_lengths (Tensor): Length tensor (B,).
            g (Optional[Tensor]): Global conditioning tensor (B, global_channels, 1).
            inverse (bool): Whether to inverse the flow.

        Returns:
            Tensor: Output tensor (B, in_channels, T).

        """
        if not inverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, inverse=inverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, inverse=inverse)
        return x


class ResidualAffineCouplingLayer(torch.nn.Module):
    """Residual affine coupling layer."""

    def __init__(
        self,
        in_channels: int = 192,
        hidden_channels: int = 192,
        kernel_size: int = 5,
        base_dilation: int = 1,
        layers: int = 5,
        stacks: int = 1,
        global_channels: int = -1,
        dropout_rate: float = 0.0,
        use_weight_norm: bool = True,
        bias: bool = True,
        use_only_mean: bool = True,
    ):
        """Initialzie ResidualAffineCouplingLayer module.

        Args:
            in_channels (int): Number of input channels.
            hidden_channels (int): Number of hidden channels.
            kernel_size (int): Kernel size for WaveNet.
            base_dilation (int): Base dilation factor for WaveNet.
            layers (int): Number of layers of WaveNet.
            stacks (int): Number of stacks of WaveNet.
            global_channels (int): Number of global channels.
            dropout_rate (float): Dropout rate.
            use_weight_norm (bool): Whether to use weight normalization in WaveNet.
            bias (bool): Whether to use bias paramters in WaveNet.
            use_only_mean (bool): Whether to estimate only mean.

        """
        assert in_channels % 2 == 0, "in_channels should be divisible by 2"
        super().__init__()
        self.half_channels = in_channels // 2
        self.use_only_mean = use_only_mean

        # define modules
        self.input_conv = torch.nn.Conv1d(self.half_channels, hidden_channels, 1,)
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
            skip_channels=hidden_channels,
            global_channels=global_channels,
            dropout_rate=dropout_rate,
            bias=bias,
            use_weight_norm=use_weight_norm,
            use_first_conv=False,
            use_last_conv=False,
            scale_residual=False,
            scale_skip_connect=True,
        )
        if use_only_mean:
            self.proj = torch.nn.Conv1d(hidden_channels, self.half_channels, 1,)
        else:
            self.proj = torch.nn.Conv1d(hidden_channels, self.half_channels * 2, 1,)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        inverse: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, in_channels, T).
            x_lengths (Tensor): Length tensor (B,).
            g (Optional[Tensor]): Global conditioning tensor (B, global_channels, 1).
            inverse (bool): Whether to inverse the flow.

        Returns:
            Tensor: Output tensor (B, in_channels, T).
            Tensor: Log-determinant tensor for NLL (B,) if not inverse.

        """
        xa, xb = x.split(x.size(1) // 2, dim=1)
        h = self.input_conv(xa) * x_mask
        h = self.encoder(h, x_mask, g=g)
        stats = self.proj(h) * x_mask
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
