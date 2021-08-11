# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Posterior encoder of VITS.

This code is based on the official implementation:
- https://github.com/jaywalnut310/vits

"""

import torch

from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet2.tts.vits.residual_block import Conv1d
from espnet2.tts.vits.wavenet import WaveNet


class PosteriorEncoder(torch.nn.Module):
    """Posterior encoder module."""

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=64,
        kernel_size=3,
        layers=30,
        stacks=3,
        global_channels=-1,
        dropout_rate=0.0,
        bias=True,
        use_weight_norm=True,
    ):
        """Initilialize PosteriorEncoder module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            hidden_channels (int): Number of hidden channels.
            kernel_size (int): Kernel size in WaveNet.
            layers (int): Number of layers of WaveNet.
            stacks (int): Number of repeat stacking of WaveNet.
            global_channels (int): Number of global conditioning channels.
            dropout_rate (float): Dropout rate.
            bias (bool): Whether to use bias parameters in conv.
            use_weight_norm (bool): Whether to apply weight norm.

        """
        super().__init__()
        self.input_conv = Conv1d(in_channels, hidden_channels, 1)
        self.encoder = WaveNet(
            in_channels=-1,
            out_channels=-1,
            kernel_size=kernel_size,
            layers=layers,
            stacks=stacks,
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
        self.output_conv = Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, in_channels, T).
            x_lengths (Tensor): Length tensor (B,).
            g (Optional[Tensor]): Global conditioning tensor (B, global_channels, 1).

        Returns:
            Tensor: VAE latent representation tensor (B, out_channels, T).
            Tensor: VAE mean tensor (B, out_channels, T).
            Tensor: VAE scale tensor (B, out_channels, T).
            Tensor: Mask tensor for input tensor (B, 1, T).

        """
        x_mask = make_non_pad_mask(x_lengths).unsqueeze(1).to(dtype=x.dtype)
        x = self.input_conv(x) * x_mask
        x = self.encoder(x, x_mask, g=g)
        stats = self.output_conv(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask
