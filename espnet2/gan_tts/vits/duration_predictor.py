# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Stochastic duration predictor modules in VITS.

This code is based on https://github.com/jaywalnut310/vits.

"""

import math
from typing import Optional

import torch
import torch.nn.functional as F

from espnet2.gan_tts.vits.flow import (
    ConvFlow,
    DilatedDepthSeparableConv,
    ElementwiseAffineFlow,
    FlipFlow,
    LogFlow,
)


class StochasticDurationPredictor(torch.nn.Module):
    """
        Stochastic duration predictor module.

    This module implements a stochastic duration predictor as described in
    `Conditional Variational Autoencoder with Adversarial Learning for
    End-to-End Text-to-Speech`_.

    .. _`Conditional Variational Autoencoder with Adversarial Learning for End-to-End
        Text-to-Speech`: https://arxiv.org/abs/2006.04558

    Attributes:
        pre (torch.nn.Conv1d): Convolutional layer for preprocessing input.
        dds (DilatedDepthSeparableConv): Dilated depth separable convolution layer.
        proj (torch.nn.Conv1d): Convolutional layer for projecting features.
        log_flow (LogFlow): Log flow for managing the flow of information.
        flows (torch.nn.ModuleList): List of flow modules for processing.
        post_pre (torch.nn.Conv1d): Convolutional layer for post-processing input.
        post_dds (DilatedDepthSeparableConv): Post-processing dilated depth
            separable convolution layer.
        post_proj (torch.nn.Conv1d): Convolutional layer for projecting post-processed
            features.
        post_flows (torch.nn.ModuleList): List of post-processing flow modules.
        global_conv (torch.nn.Conv1d, optional): Convolutional layer for global
            conditioning if global_channels > 0.

    Args:
        channels (int): Number of channels.
        kernel_size (int): Kernel size.
        dropout_rate (float): Dropout rate.
        flows (int): Number of flows.
        dds_conv_layers (int): Number of conv layers in DDS conv.
        global_channels (int): Number of global conditioning channels.

    Examples:
        >>> predictor = StochasticDurationPredictor()
        >>> x = torch.randn(2, 192, 50)  # Example input tensor
        >>> x_mask = torch.ones(2, 1, 50)  # Example mask tensor
        >>> duration = torch.randn(2, 1, 50)  # Example duration tensor
        >>> output = predictor(x, x_mask, w=duration)

    Raises:
        AssertionError: If `inverse` is False and `w` is None in the forward method.
    """

    def __init__(
        self,
        channels: int = 192,
        kernel_size: int = 3,
        dropout_rate: float = 0.5,
        flows: int = 4,
        dds_conv_layers: int = 3,
        global_channels: int = -1,
    ):
        """Initialize StochasticDurationPredictor module.

        Args:
            channels (int): Number of channels.
            kernel_size (int): Kernel size.
            dropout_rate (float): Dropout rate.
            flows (int): Number of flows.
            dds_conv_layers (int): Number of conv layers in DDS conv.
            global_channels (int): Number of global conditioning channels.

        """
        super().__init__()

        self.pre = torch.nn.Conv1d(channels, channels, 1)
        self.dds = DilatedDepthSeparableConv(
            channels,
            kernel_size,
            layers=dds_conv_layers,
            dropout_rate=dropout_rate,
        )
        self.proj = torch.nn.Conv1d(channels, channels, 1)

        self.log_flow = LogFlow()
        self.flows = torch.nn.ModuleList()
        self.flows += [ElementwiseAffineFlow(2)]
        for i in range(flows):
            self.flows += [
                ConvFlow(
                    2,
                    channels,
                    kernel_size,
                    layers=dds_conv_layers,
                )
            ]
            self.flows += [FlipFlow()]

        self.post_pre = torch.nn.Conv1d(1, channels, 1)
        self.post_dds = DilatedDepthSeparableConv(
            channels,
            kernel_size,
            layers=dds_conv_layers,
            dropout_rate=dropout_rate,
        )
        self.post_proj = torch.nn.Conv1d(channels, channels, 1)
        self.post_flows = torch.nn.ModuleList()
        self.post_flows += [ElementwiseAffineFlow(2)]
        for i in range(flows):
            self.post_flows += [
                ConvFlow(
                    2,
                    channels,
                    kernel_size,
                    layers=dds_conv_layers,
                )
            ]
            self.post_flows += [FlipFlow()]

        if global_channels > 0:
            self.global_conv = torch.nn.Conv1d(global_channels, channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        w: Optional[torch.Tensor] = None,
        g: Optional[torch.Tensor] = None,
        inverse: bool = False,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """
                Calculate forward propagation.

        This method performs the forward pass for the Stochastic Duration Predictor.
        It computes the negative log-likelihood (NLL) or log-duration tensor based
        on the provided input tensors and optional parameters.

        Args:
            x (Tensor): Input tensor with shape (B, channels, T_text).
            x_mask (Tensor): Mask tensor with shape (B, 1, T_text).
            w (Optional[Tensor]): Duration tensor with shape (B, 1, T_text).
                Required when `inverse` is False.
            g (Optional[Tensor]): Global conditioning tensor with shape (B, channels, 1).
            inverse (bool): Whether to perform the inverse operation on the flow.
                Defaults to False.
            noise_scale (float): Scale for the noise added to the latent space.
                Defaults to 1.0.

        Returns:
            Tensor: If `inverse` is False, returns a negative log-likelihood (NLL)
            tensor with shape (B,). If `inverse` is True, returns a log-duration
            tensor with shape (B, 1, T_text).

        Examples:
            >>> model = StochasticDurationPredictor()
            >>> x = torch.randn(5, 192, 10)  # Example input tensor
            >>> x_mask = torch.ones(5, 1, 10)  # Example mask tensor
            >>> w = torch.randn(5, 1, 10)  # Example duration tensor
            >>> output = model.forward(x, x_mask, w)  # Forward pass
        """
        x = x.detach()  # stop gradient
        x = self.pre(x)
        if g is not None:
            x = x + self.global_conv(g.detach())  # stop gradient
        x = self.dds(x, x_mask)
        x = self.proj(x) * x_mask

        if not inverse:
            assert w is not None, "w must be provided."
            h_w = self.post_pre(w)
            h_w = self.post_dds(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = (
                torch.randn(
                    w.size(0),
                    2,
                    w.size(2),
                ).to(device=x.device, dtype=x.dtype)
                * x_mask
            )
            z_q = e_q
            logdet_tot_q = 0.0
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum(
                (F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2]
            )
            logq = (
                torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q**2)) * x_mask, [1, 2])
                - logdet_tot_q
            )

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in self.flows:
                z, logdet = flow(z, x_mask, g=x, inverse=inverse)
                logdet_tot = logdet_tot + logdet
            nll = (
                torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2])
                - logdet_tot
            )
            return nll + logq  # (B,)
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            z = (
                torch.randn(
                    x.size(0),
                    2,
                    x.size(2),
                ).to(device=x.device, dtype=x.dtype)
                * noise_scale
            )
            for flow in flows:
                z = flow(z, x_mask, g=x, inverse=inverse)
            z0, z1 = z.split(1, 1)
            logw = z0
            return logw
