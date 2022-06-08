# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""HiFiGAN Residual block modules.

This code is modified from https://github.com/kan-bayashi/ParallelWaveGAN.

"""

from typing import Any, Dict, List

import torch


class ResidualBlock(torch.nn.Module):
    """Residual block module in HiFiGAN."""

    def __init__(
        self,
        kernel_size: int = 3,
        channels: int = 512,
        dilations: List[int] = [1, 3, 5],
        bias: bool = True,
        use_additional_convs: bool = True,
        nonlinear_activation: str = "LeakyReLU",
        nonlinear_activation_params: Dict[str, Any] = {"negative_slope": 0.1},
    ):
        """Initialize ResidualBlock module.

        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            channels (int): Number of channels for convolution layer.
            dilations (List[int]): List of dilation factors.
            use_additional_convs (bool): Whether to use additional convolution layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (Dict[str, Any]): Hyperparameters for activation
                function.

        """
        super().__init__()
        self.use_additional_convs = use_additional_convs
        self.convs1 = torch.nn.ModuleList()
        if use_additional_convs:
            self.convs2 = torch.nn.ModuleList()
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        for dilation in dilations:
            self.convs1 += [
                torch.nn.Sequential(
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                    torch.nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation,
                        bias=bias,
                        padding=(kernel_size - 1) // 2 * dilation,
                    ),
                )
            ]
            if use_additional_convs:
                self.convs2 += [
                    torch.nn.Sequential(
                        getattr(torch.nn, nonlinear_activation)(
                            **nonlinear_activation_params
                        ),
                        torch.nn.Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            1,
                            dilation=1,
                            bias=bias,
                            padding=(kernel_size - 1) // 2,
                        ),
                    )
                ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, channels, T).

        Returns:
            Tensor: Output tensor (B, channels, T).

        """
        for idx in range(len(self.convs1)):
            xt = self.convs1[idx](x)
            if self.use_additional_convs:
                xt = self.convs2[idx](xt)
            x = xt + x
        return x
