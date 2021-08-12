# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Stochastic duration predictor modules of VITS.

This code is based on https://github.com/jaywalnut310/vits.

"""

import math

import torch
import torch.nn.functional as F

from espnet2.tts.vits.flow import ConvFlow
from espnet2.tts.vits.flow import DilatedDepthSeparableConv
from espnet2.tts.vits.flow import ElementwiseAffineFlow
from espnet2.tts.vits.flow import Flip
from espnet2.tts.vits.flow import LogFlow


class StochasticDurationPredictor(torch.nn.Module):
    """Stochastic duration predictor module."""

    def __init__(
        self,
        channels=192,
        kernel_size=3,
        dropout_rate=0.5,
        flows=4,
        dds_conv_layers=3,
        global_channels=-1,
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

        self.conv1 = torch.nn.Conv1d(channels, channels, 1)
        self.dds_conv = DilatedDepthSeparableConv(
            channels,
            kernel_size,
            layers=dds_conv_layers,
            dropout_rate=dropout_rate,
        )
        self.conv2 = torch.nn.Conv1d(channels, channels, 1)

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
            self.flows += [Flip()]

        self.post_conv1 = torch.nn.Conv1d(1, channels, 1)
        self.post_dds_conv = DilatedDepthSeparableConv(
            channels,
            kernel_size,
            layers=dds_conv_layers,
            dropout_rate=dropout_rate,
        )
        self.post_conv2 = torch.nn.Conv1d(channels, channels, 1)
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
            self.post_flows += [Flip()]

        if global_channels > 0:
            self.global_conv = torch.nn.Conv1d(global_channels, channels, 1)

    def forward(self, x, x_mask, w=None, g=None, inverse=False, noise_scale=1.0):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, channels, T).
            x_mask (Tensor): Mask tensor (B, 1, T).
            w (Optional[Tensor]): Duration tensor (B, 1, T).
            g (Optional[Tensor]): Global conditioning tensor (B, channels, 1)
            inverse (bool): Whether to inverse the flow.
            noise_scale (float): Noise scale value.

        Returns:
            Tensor: Negative lower bound tensor (B,) if not inverse.

        """
        x = self.conv1(x.detach())  # stop gradient
        if g is not None:
            x = x + self.global_conv(g.detach())  # stop gradient
        x = self.dds_conv(x, x_mask)
        x = self.conv2(x) * x_mask

        if not inverse:
            assert w is not None, "w must be provided."
            h_w = self.post_conv1(w)
            h_w = self.post_dds_conv(h_w, x_mask)
            h_w = self.post_conv2(h_w) * x_mask
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
                torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q ** 2)) * x_mask, [1, 2])
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
                torch.sum(0.5 * (math.log(2 * math.pi) + (z ** 2)) * x_mask, [1, 2])
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
