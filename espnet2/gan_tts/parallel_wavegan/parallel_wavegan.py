# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Parallel WaveGAN Modules.

This code is modified from https://github.com/kan-bayashi/ParallelWaveGAN.

"""

import logging
import math
from typing import Any, Dict, Optional

import numpy as np
import torch

from espnet2.gan_tts.parallel_wavegan import upsample
from espnet2.gan_tts.wavenet.residual_block import Conv1d, Conv1d1x1, ResidualBlock


class ParallelWaveGANGenerator(torch.nn.Module):
    """Parallel WaveGAN Generator module."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        layers: int = 30,
        stacks: int = 3,
        residual_channels: int = 64,
        gate_channels: int = 128,
        skip_channels: int = 64,
        aux_channels: int = 80,
        aux_context_window: int = 2,
        dropout_rate: float = 0.0,
        bias: bool = True,
        use_weight_norm: bool = True,
        upsample_conditional_features: bool = True,
        upsample_net: str = "ConvInUpsampleNetwork",
        upsample_params: Dict[str, Any] = {"upsample_scales": [4, 4, 4, 4]},
    ):
        """Initialize ParallelWaveGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of dilated convolution.
            layers (int): Number of residual block layers.
            stacks (int): Number of stacks i.e., dilation cycles.
            residual_channels (int): Number of channels in residual conv.
            gate_channels (int):  Number of channels in gated conv.
            skip_channels (int): Number of channels in skip conv.
            aux_channels (int): Number of channels for auxiliary feature conv.
            aux_context_window (int): Context window size for auxiliary feature.
            dropout_rate (float): Dropout rate. 0.0 means no dropout applied.
            bias (bool): Whether to use bias parameter in conv layer.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            upsample_conditional_features (bool): Whether to use upsampling network.
            upsample_net (str): Upsampling network architecture.
            upsample_params (Dict[str, Any]): Upsampling network parameters.

        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aux_channels = aux_channels
        self.aux_context_window = aux_context_window
        self.layers = layers
        self.stacks = stacks
        self.kernel_size = kernel_size

        # check the number of layers and stacks
        assert layers % stacks == 0
        layers_per_stack = layers // stacks

        # define first convolution
        self.first_conv = Conv1d1x1(in_channels, residual_channels, bias=True)

        # define conv + upsampling network
        if upsample_conditional_features:
            if upsample_net == "ConvInUpsampleNetwork":
                upsample_params.update(
                    {
                        "aux_channels": aux_channels,
                        "aux_context_window": aux_context_window,
                    }
                )
            self.upsample_net = getattr(upsample, upsample_net)(**upsample_params)
            self.upsample_factor = int(np.prod(upsample_params["upsample_scales"]))
        else:
            self.upsample_net = None
            self.upsample_factor = out_channels

        # define residual blocks
        self.conv_layers = torch.nn.ModuleList()
        for layer in range(layers):
            dilation = 2 ** (layer % layers_per_stack)
            conv = ResidualBlock(
                kernel_size=kernel_size,
                residual_channels=residual_channels,
                gate_channels=gate_channels,
                skip_channels=skip_channels,
                aux_channels=aux_channels,
                dilation=dilation,
                dropout_rate=dropout_rate,
                bias=bias,
                scale_residual=True,
            )
            self.conv_layers += [conv]

        # define output layers
        self.last_conv_layers = torch.nn.ModuleList(
            [
                torch.nn.ReLU(),
                Conv1d1x1(skip_channels, skip_channels, bias=True),
                torch.nn.ReLU(),
                Conv1d1x1(skip_channels, out_channels, bias=True),
            ]
        )

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # NOTE(kan-bayashi): register pre hook function for the compatibility with
        #   parallel_wavegan repo
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def forward(
        self, c: torch.Tensor, z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            c (Tensor): Local conditioning auxiliary features (B, C ,T_feats).
            z (Tensor): Input noise signal (B, 1, T_wav).

        Returns:
            Tensor: Output tensor (B, out_channels, T_wav)

        """
        if z is None:
            b, _, t = c.size()
            z = torch.randn(b, 1, t * self.upsample_factor).to(
                device=c.device, dtype=c.dtype
            )

        # perform upsampling
        if self.upsample_net is not None:
            c = self.upsample_net(c)
            assert c.size(-1) == z.size(-1)

        # encode to hidden representation
        x = self.first_conv(z)
        skips = 0
        for f in self.conv_layers:
            x, h = f(x=x, x_mask=None, c=c)
            skips += h
        skips *= math.sqrt(1.0 / len(self.conv_layers))

        # apply final layers
        x = skips
        for f in self.last_conv_layers:
            x = f(x)

        return x

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
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    @staticmethod
    def _get_receptive_field_size(
        layers, stacks, kernel_size, dilation=lambda x: 2**x
    ):
        assert layers % stacks == 0
        layers_per_cycle = layers // stacks
        dilations = [dilation(i % layers_per_cycle) for i in range(layers)]
        return (kernel_size - 1) * sum(dilations) + 1

    @property
    def receptive_field_size(self):
        """Return receptive field size."""
        return self._get_receptive_field_size(
            self.layers, self.stacks, self.kernel_size
        )

    def inference(
        self, c: torch.Tensor, z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Perform inference.

        Args:
            c (Tensor): Local conditioning auxiliary features (T_feats ,C).
            z (Optional[Tensor]): Input noise signal (T_wav, 1).

        Returns:
            Tensor: Output tensor (T_wav, out_channels)

        """
        if z is not None:
            z = z.transpose(1, 0).unsqueeze(0)
        c = c.transpose(1, 0).unsqueeze(0)
        return self.forward(c, z).squeeze(0).transpose(1, 0)

    def _load_state_dict_pre_hook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Apply pre hook function before loading state dict."""
        keys = list(state_dict.keys())
        for k in keys:
            if "conv1x1_skip" in k.replace(prefix, ""):
                v_skip = state_dict.pop(k)
                v_out = state_dict[k.replace("skip", "out")]
                state_dict[k.replace("skip", "out")] = torch.cat([v_out, v_skip], dim=0)


class ParallelWaveGANDiscriminator(torch.nn.Module):
    """Parallel WaveGAN Discriminator module."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        layers: int = 10,
        conv_channels: int = 64,
        dilation_factor: int = 1,
        nonlinear_activation: str = "LeakyReLU",
        nonlinear_activation_params: Dict[str, Any] = {"negative_slope": 0.2},
        bias: bool = True,
        use_weight_norm: bool = True,
    ):
        """Initialize ParallelWaveGANDiscriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Number of output channels.
            layers (int): Number of conv layers.
            conv_channels (int): Number of chnn layers.
            dilation_factor (int): Dilation factor. For example, if dilation_factor = 2,
                the dilation will be 2, 4, 8, ..., and so on.
            nonlinear_activation (str): Nonlinear function after each conv.
            nonlinear_activation_params (Dict[str, Any]): Nonlinear function parameters
            bias (bool): Whether to use bias parameter in conv.
            use_weight_norm (bool) Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
        assert dilation_factor > 0, "Dilation factor must be > 0."
        self.conv_layers = torch.nn.ModuleList()
        conv_in_channels = in_channels
        for i in range(layers - 1):
            if i == 0:
                dilation = 1
            else:
                dilation = i if dilation_factor == 1 else dilation_factor**i
                conv_in_channels = conv_channels
            padding = (kernel_size - 1) // 2 * dilation
            conv_layer = [
                Conv1d(
                    conv_in_channels,
                    conv_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                    bias=bias,
                ),
                getattr(torch.nn, nonlinear_activation)(
                    inplace=True, **nonlinear_activation_params
                ),
            ]
            self.conv_layers += conv_layer
        padding = (kernel_size - 1) // 2
        last_conv_layer = Conv1d(
            conv_in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )
        self.conv_layers += [last_conv_layer]

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            Tensor: Output tensor (B, 1, T).

        """
        for f in self.conv_layers:
            x = f(x)
        return x

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m: torch.nn.Module):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m: torch.nn.Module):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)
