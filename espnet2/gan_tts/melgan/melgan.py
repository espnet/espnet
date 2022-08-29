# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""MelGAN Modules.

This code is modified from https://github.com/kan-bayashi/ParallelWaveGAN.

"""

import logging
from typing import Any, Dict, List

import numpy as np
import torch

from espnet2.gan_tts.melgan.residual_stack import ResidualStack


class MelGANGenerator(torch.nn.Module):
    """MelGAN generator module."""

    def __init__(
        self,
        in_channels: int = 80,
        out_channels: int = 1,
        kernel_size: int = 7,
        channels: int = 512,
        bias: bool = True,
        upsample_scales: List[int] = [8, 8, 2, 2],
        stack_kernel_size: int = 3,
        stacks: int = 3,
        nonlinear_activation: str = "LeakyReLU",
        nonlinear_activation_params: Dict[str, Any] = {"negative_slope": 0.2},
        pad: str = "ReflectionPad1d",
        pad_params: Dict[str, Any] = {},
        use_final_nonlinear_activation: bool = True,
        use_weight_norm: bool = True,
    ):
        """Initialize MelGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            channels (int): Initial number of channels for conv layer.
            bias (bool): Whether to add bias parameter in convolution layers.
            upsample_scales (List[int]): List of upsampling scales.
            stack_kernel_size (int): Kernel size of dilated conv layers in residual
                stack.
            stacks (int): Number of stacks in a single residual stack.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (Dict[str, Any]): Hyperparameters for activation
                function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (Dict[str, Any]): Hyperparameters for padding function.
            use_final_nonlinear_activation (torch.nn.Module): Activation function for
                the final layer.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super().__init__()

        # check hyper parameters is valid
        assert channels >= np.prod(upsample_scales)
        assert channels % (2 ** len(upsample_scales)) == 0
        assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."

        # add initial layer
        layers = []
        layers += [
            getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params),
            torch.nn.Conv1d(in_channels, channels, kernel_size, bias=bias),
        ]

        self.upsample_factor = int(np.prod(upsample_scales) * out_channels)
        for i, upsample_scale in enumerate(upsample_scales):
            # add upsampling layer
            layers += [
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)
            ]
            layers += [
                torch.nn.ConvTranspose1d(
                    channels // (2 ** i),
                    channels // (2 ** (i + 1)),
                    upsample_scale * 2,
                    stride=upsample_scale,
                    padding=upsample_scale // 2 + upsample_scale % 2,
                    output_padding=upsample_scale % 2,
                    bias=bias,
                )
            ]

            # add residual stack
            for j in range(stacks):
                layers += [
                    ResidualStack(
                        kernel_size=stack_kernel_size,
                        channels=channels // (2 ** (i + 1)),
                        dilation=stack_kernel_size ** j,
                        bias=bias,
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                        pad=pad,
                        pad_params=pad_params,
                    )
                ]

        # add final layer
        layers += [
            getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)
        ]
        layers += [
            getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params),
            torch.nn.Conv1d(
                channels // (2 ** (i + 1)), out_channels, kernel_size, bias=bias
            ),
        ]
        if use_final_nonlinear_activation:
            layers += [torch.nn.Tanh()]

        # define the model as a single function
        self.melgan = torch.nn.Sequential(*layers)

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, channels, T).

        Returns:
            Tensor: Output tensor (B, 1, T ** prod(upsample_scales)).

        """
        return self.melgan(c)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m: torch.nn.Module):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m: torch.nn.Module):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows official implementation manner.
        https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py

        """

        def _reset_parameters(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                m.weight.data.normal_(0.0, 0.02)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)

    def inference(self, c: torch.Tensor) -> torch.Tensor:
        """Perform inference.

        Args:
            c (Tensor): Input tensor (T, in_channels).

        Returns:
            Tensor: Output tensor (T ** prod(upsample_scales), out_channels).

        """
        c = self.melgan(c.transpose(1, 0).unsqueeze(0))
        return c.squeeze(0).transpose(1, 0)


class MelGANDiscriminator(torch.nn.Module):
    """MelGAN discriminator module."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_sizes: List[int] = [5, 3],
        channels: int = 16,
        max_downsample_channels: int = 1024,
        bias: bool = True,
        downsample_scales: List[int] = [4, 4, 4, 4],
        nonlinear_activation: str = "LeakyReLU",
        nonlinear_activation_params: Dict[str, Any] = {"negative_slope": 0.2},
        pad: str = "ReflectionPad1d",
        pad_params: Dict[str, Any] = {},
    ):
        """Initilize MelGANDiscriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_sizes (List[int]): List of two kernel sizes. The prod will be used
                for the first conv layer, and the first and the second kernel sizes
                will be used for the last two layers. For example if kernel_sizes =
                [5, 3], the first layer kernel size will be 5 * 3 = 15, the last two
                layers' kernel size will be 5 and 3, respectively.
            channels (int): Initial number of channels for conv layer.
            max_downsample_channels (int): Maximum number of channels for downsampling
                layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (List[int]): List of downsampling scales.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (Dict[str, Any]): Hyperparameters for activation
                function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (Dict[str, Any]): Hyperparameters for padding function.

        """
        super().__init__()
        self.layers = torch.nn.ModuleList()

        # check kernel size is valid
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1
        assert kernel_sizes[1] % 2 == 1

        # add first layer
        self.layers += [
            torch.nn.Sequential(
                getattr(torch.nn, pad)((np.prod(kernel_sizes) - 1) // 2, **pad_params),
                torch.nn.Conv1d(
                    in_channels, channels, np.prod(kernel_sizes), bias=bias
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]

        # add downsample layers
        in_chs = channels
        for downsample_scale in downsample_scales:
            out_chs = min(in_chs * downsample_scale, max_downsample_channels)
            self.layers += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chs,
                        out_chs,
                        kernel_size=downsample_scale * 10 + 1,
                        stride=downsample_scale,
                        padding=downsample_scale * 5,
                        groups=in_chs // 4,
                        bias=bias,
                    ),
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                )
            ]
            in_chs = out_chs

        # add final layers
        out_chs = min(in_chs * 2, max_downsample_channels)
        self.layers += [
            torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_chs,
                    out_chs,
                    kernel_sizes[0],
                    padding=(kernel_sizes[0] - 1) // 2,
                    bias=bias,
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]
        self.layers += [
            torch.nn.Conv1d(
                out_chs,
                out_channels,
                kernel_sizes[1],
                padding=(kernel_sizes[1] - 1) // 2,
                bias=bias,
            ),
        ]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            List[Tensor]: List of output tensors of each layer.

        """
        outs = []
        for f in self.layers:
            x = f(x)
            outs += [x]

        return outs


class MelGANMultiScaleDiscriminator(torch.nn.Module):
    """MelGAN multi-scale discriminator module."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        scales: int = 3,
        downsample_pooling: str = "AvgPool1d",
        # follow the official implementation setting
        downsample_pooling_params: Dict[str, Any] = {
            "kernel_size": 4,
            "stride": 2,
            "padding": 1,
            "count_include_pad": False,
        },
        kernel_sizes: List[int] = [5, 3],
        channels: int = 16,
        max_downsample_channels: int = 1024,
        bias: bool = True,
        downsample_scales: List[int] = [4, 4, 4, 4],
        nonlinear_activation: str = "LeakyReLU",
        nonlinear_activation_params: Dict[str, Any] = {"negative_slope": 0.2},
        pad: str = "ReflectionPad1d",
        pad_params: Dict[str, Any] = {},
        use_weight_norm: bool = True,
    ):
        """Initilize MelGANMultiScaleDiscriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            scales (int): Number of multi-scales.
            downsample_pooling (str): Pooling module name for downsampling of the
                inputs.
            downsample_pooling_params (Dict[str, Any]): Parameters for the above
                pooling module.
            kernel_sizes (List[int]): List of two kernel sizes. The sum will be used
                for the first conv layer, and the first and the second kernel sizes
                will be used for the last two layers.
            channels (int): Initial number of channels for conv layer.
            max_downsample_channels (int): Maximum number of channels for downsampling
                layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (List[int]): List of downsampling scales.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (Dict[str, Any]): Hyperparameters for activation
                function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (Dict[str, Any]): Hyperparameters for padding function.
            use_weight_norm (bool): Whether to use weight norm.

        """
        super().__init__()
        self.discriminators = torch.nn.ModuleList()

        # add discriminators
        for _ in range(scales):
            self.discriminators += [
                MelGANDiscriminator(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_sizes=kernel_sizes,
                    channels=channels,
                    max_downsample_channels=max_downsample_channels,
                    bias=bias,
                    downsample_scales=downsample_scales,
                    nonlinear_activation=nonlinear_activation,
                    nonlinear_activation_params=nonlinear_activation_params,
                    pad=pad,
                    pad_params=pad_params,
                )
            ]
        self.pooling = getattr(torch.nn, downsample_pooling)(
            **downsample_pooling_params
        )

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            List[List[Tensor]]: List of list of each discriminator outputs, which
                consists of each layer output tensors.

        """
        outs = []
        for f in self.discriminators:
            outs += [f(x)]
            x = self.pooling(x)

        return outs

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m: torch.nn.Module):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m: torch.nn.Module):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows official implementation manner.
        https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py

        """

        def _reset_parameters(m: torch.nn.Module):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                m.weight.data.normal_(0.0, 0.02)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)
