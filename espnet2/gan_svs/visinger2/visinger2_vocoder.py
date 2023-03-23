# Copyright 2023 Yifeng Yu
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""VISinger2 HiFi-GAN Modules.

This code is based on https://github.com/zhangyongmao/VISinger2

"""

import copy
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import math

from parallel_wavegan.layers import CausalConv1d
from parallel_wavegan.layers import CausalConvTranspose1d
from parallel_wavegan.layers import HiFiGANResidualBlock as ResidualBlock
from parallel_wavegan.utils import read_hdf5
from espnet2.gan_tts.hifigan import (
    HiFiGANMultiPeriodDiscriminator,
    HiFiGANMultiScaleDiscriminator,
    HiFiGANMultiScaleMultiPeriodDiscriminator,
    HiFiGANPeriodDiscriminator,
    HiFiGANScaleDiscriminator,
)

from espnet2.gan_svs.visinger2.ddsp import (
    scale_function,
    remove_above_nyquist,
    upsample,
)


class VISinger2VocoderGenerator(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 80,
        out_channels: int = 1,
        channels: int = 512,
        global_channels: int = -1,
        kernel_size: int = 7,
        upsample_scales: List[int] = [8, 8, 2, 2],
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilations: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        n_harmonic: int = 64,
        use_additional_convs: bool = True,
        bias: bool = True,
        nonlinear_activation: str = "LeakyReLU",
        nonlinear_activation_params: Dict[str, Any] = {"negative_slope": 0.1},
        use_weight_norm: bool = True,
    ):
        """Initialize HiFiGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            channels (int): Number of hidden representation channels.
            global_channels (int): Number of global conditioning channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            upsample_scales (List[int]): List of upsampling scales.
            upsample_kernel_sizes (List[int]): List of kernel sizes for upsample layers.
            resblock_kernel_sizes (List[int]): List of kernel sizes for residual blocks.
            resblock_dilations (List[List[int]]): List of list of dilations for residual
                blocks.
            use_additional_convs (bool): Whether to use additional conv layers in
                residual blocks.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (Dict[str, Any]): Hyperparameters for activation
                function.
            use_weight_norm (bool): Whether to use weight norm. If set to true, it will
                be applied to all of the conv layers.
            # TODO(Yifeng): fix comments

        """
        super().__init__()

        # check hyperparameters are valid
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        assert len(upsample_scales) == len(upsample_kernel_sizes)
        assert len(resblock_dilations) == len(resblock_kernel_sizes)

        # define modules
        self.upsample_factor = int(np.prod(upsample_scales) * out_channels)
        self.num_upsamples = len(upsample_kernel_sizes)
        self.num_blocks = len(resblock_kernel_sizes)
        self.input_conv = torch.nn.Conv1d(
            in_channels,
            channels,
            kernel_size,
            1,
            padding=(kernel_size - 1) // 2,
        )
        self.upsample_scales = upsample_scales

        self.downs = torch.nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_scales, upsample_kernel_sizes)):
            i = self.num_upsamples - 1 - i
            u = upsample_scales[i]
            k = upsample_kernel_sizes[i]
            down = torch.nn.Conv1d(
                n_harmonic + 2,
                n_harmonic + 2,
                k,
                u,
                padding=k // 2,
            )
            self.downs.append(down)

        self.blocks_downs = torch.nn.ModuleList()
        for i in range(len(self.downs)):
            j = self.num_upsamples - 1 - i

            self.blocks_downs += [
                ResidualBlock(
                    kernel_size=3,
                    channels=n_harmonic + 2,
                    dilations=(1, 3),
                    bias=bias,
                    use_additional_convs=False,
                    nonlinear_activation=nonlinear_activation,
                    nonlinear_activation_params=nonlinear_activation_params,
                )
            ]

        self.concat_pre = torch.nn.Conv1d(
            channels + n_harmonic + 2,
            channels,
            3,
            1,
            padding=1,
        )
        self.concat_conv = torch.nn.ModuleList()
        for i in range(self.num_upsamples):
            ch = channels // (2 ** (i + 1))
            self.concat_conv.append(
                torch.nn.Conv1d(ch + n_harmonic + 2, ch, 3, 1, padding=1, bias=bias)
            )

        self.upsamples = torch.nn.ModuleList()
        self.blocks = torch.nn.ModuleList()
        for i in range(len(upsample_kernel_sizes)):
            assert upsample_kernel_sizes[i] == 2 * upsample_scales[i]
            self.upsamples += [
                torch.nn.Sequential(
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                    torch.nn.ConvTranspose1d(
                        channels // (2**i),
                        channels // (2 ** (i + 1)),
                        upsample_kernel_sizes[i],
                        upsample_scales[i],
                        padding=upsample_scales[i] // 2 + upsample_scales[i] % 2,
                        output_padding=upsample_scales[i] % 2,
                    ),
                )
            ]
            for j in range(len(resblock_kernel_sizes)):
                self.blocks += [
                    ResidualBlock(
                        kernel_size=resblock_kernel_sizes[j],
                        channels=channels // (2 ** (i + 1)),
                        dilations=resblock_dilations[j],
                        bias=bias,
                        use_additional_convs=use_additional_convs,
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                    )
                ]
        self.output_conv = torch.nn.Sequential(
            # NOTE(kan-bayashi): follow official implementation but why
            #   using different slope parameter here? (0.1 vs. 0.01)
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(
                channels // (2 ** (i + 1)),
                out_channels,
                kernel_size,
                1,
                padding=(kernel_size - 1) // 2,
            ),
            torch.nn.Tanh(),
        )
        if global_channels > 0:
            self.global_conv = torch.nn.Conv1d(global_channels, channels, 1)

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, c, ddsp, g: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, in_channels, T).
            g (Optional[Tensor]): Global conditioning tensor (B, global_channels, 1).

        Returns:
            Tensor: Output tensor (B, out_channels, T).

        """
        c = self.input_conv(c)
        if g is not None:
            c = c + self.global_conv(g)

        se = ddsp
        res_features = [se]
        for i in range(self.num_upsamples):
            in_size = se.size(2)
            se = self.downs[i](se)
            se = self.blocks_downs[i](se)
            up_rate = self.upsample_scales[self.num_upsamples - 1 - i]
            se = se[:, :, : in_size // up_rate]
            res_features.append(se)

        c = torch.cat([c, se], 1)
        c = self.concat_pre(c)

        for i in range(self.num_upsamples):
            in_size = c.size(2)
            c = self.upsamples[i](c)
            c = c[:, :, : in_size * self.upsample_scales[i]]

            c = torch.cat([c, res_features[self.num_upsamples - 1 - i]], 1)
            c = self.concat_conv[i](c)

            cs = 0.0  # initialize
            for j in range(self.num_blocks):
                cs += self.blocks[i * self.num_blocks + j](c)
            c = cs / self.num_blocks
        c = self.output_conv(c)

        return c

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows the official implementation manner.
        https://github.com/jik876/hifi-gan/blob/master/models.py

        """

        def _reset_parameters(m: torch.nn.Module):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                m.weight.data.normal_(0.0, 0.01)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)

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


class Generator_Harm(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int = 192,
        n_harmonic: int = 64,
        kernel_size: int = 3,
        padding: int = 1,
        p_dropout: float = 0.1,
        sample_rate: int = 22050,
        hop_size: int = 256,
    ):
        super().__init__()

        self.prenet = torch.nn.Sequential(
            torch.nn.Conv1d(
                hidden_channels, hidden_channels, kernel_size, padding=padding
            ),
            torch.nn.ReLU(),
        )

        self.net = ConvReluNorm(
            hidden_channels,
            hidden_channels,
            hidden_channels,
            kernel_size,
            8,
            p_dropout,
        )

        self.postnet = torch.nn.Sequential(
            torch.nn.Conv1d(
                hidden_channels, n_harmonic + 1, kernel_size, padding=padding
            ),
            torch.nn.ReLU(),
        )

        self.sample_rate = sample_rate
        self.hop_size = hop_size

    def forward(self, f0, harm, mask):
        pitch = f0.transpose(1, 2)
        harm = self.prenet(harm)

        harm = self.net(harm) * mask

        harm = self.postnet(harm)
        harm = harm.transpose(1, 2)
        param = harm

        param = scale_function(param)
        total_amp = param[..., :1]
        amplitudes = param[..., 1:]
        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sample_rate,
        )
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp

        amplitudes = upsample(amplitudes, self.hop_size)
        pitch = upsample(pitch, self.hop_size)

        n_harmonic = amplitudes.shape[-1]
        omega = torch.cumsum(2 * math.pi * pitch / self.sample_rate, 1)
        omegas = omega * torch.arange(1, n_harmonic + 1).to(omega)
        signal_harmonics = torch.sin(omegas) * amplitudes
        signal_harmonics = signal_harmonics.transpose(1, 2)
        return signal_harmonics


class Generator_Noise(torch.nn.Module):
    def __init__(
        self,
        win_length: int = 1024,
        hop_length: int = 256,
        n_fft: int = 1024,
        hidden_channels: int = 192,
        kernel_size: int = 3,
        padding: int = 1,
        p_dropout: float = 0.1,
    ):
        super().__init__()
        self.win_size = win_length if win_length is not None else n_fft
        self.hop_size = hop_length
        self.fft_size = n_fft
        self.istft_pre = torch.nn.Conv1d(
            hidden_channels, hidden_channels, kernel_size, padding=padding
        )

        self.net = ConvReluNorm(
            hidden_channels,
            hidden_channels,
            hidden_channels,
            kernel_size,
            8,
            p_dropout,
        )

        self.istft_amplitude = torch.nn.Conv1d(
            hidden_channels, self.fft_size // 2 + 1, 1, padding
        )
        self.window = torch.hann_window(self.win_size)

    def forward(self, x, mask):
        istft_x = x
        istft_x = self.istft_pre(istft_x)

        istft_x = self.net(istft_x) * mask

        amp = self.istft_amplitude(istft_x).unsqueeze(-1)
        phase = (torch.rand(amp.shape) * 2 * 3.14 - 3.14).to(amp)

        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        # spec = torch.cat([real, imag], 3)
        spec = torch.stack([real, imag], dim=-1)  # Change to stack
        spec_complex = torch.view_as_complex(
            spec.squeeze(-2)
        )  # Convert to complex tensor
        istft_x = torch.istft(
            spec_complex,
            self.fft_size,
            self.hop_size,
            self.win_size,
            self.window.to(amp),
            True,
            length=x.shape[2] * self.hop_size,
            return_complex=False,
        )

        return istft_x.unsqueeze(1)


class MultiFrequencyDiscriminator(torch.nn.Module):
    def __init__(
        self,
        hop_lengths=[128, 256, 512],
        hidden_channels=[256, 512, 512],
        domain="double",
        mel_scale=True,
    ):
        super().__init__()

        self.stfts = torch.nn.ModuleList(
            [
                TorchSTFT(
                    fft_size=x * 4,
                    hop_size=x,
                    win_size=x * 4,
                    normalized=True,
                    domain=domain,
                    mel_scale=mel_scale,
                )
                for x in hop_lengths
            ]
        )

        self.domain = domain
        if domain == "double":
            self.discriminators = torch.nn.ModuleList(
                [
                    BaseFrequenceDiscriminator(2, c)
                    for x, c in zip(hop_lengths, hidden_channels)
                ]
            )
        else:
            self.discriminators = torch.nn.ModuleList(
                [
                    BaseFrequenceDiscriminator(1, c)
                    for x, c in zip(hop_lengths, hidden_channels)
                ]
            )

    def forward(self, x):
        scores, feats = list(), list()
        for stft, layer in zip(self.stfts, self.discriminators):
            # print(stft)
            mag, phase = stft.transform(x.squeeze())
            if self.domain == "double":
                mag = torch.stack(torch.chunk(mag, 2, dim=1), dim=1)
            else:
                mag = mag.unsqueeze(1)

            score, feat = layer(mag)
            scores.append(score)
            feats.append(feat)
        return scores, feats


class BaseFrequenceDiscriminator(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=512):
        super().__init__()

        self.discriminator = torch.nn.ModuleList()
        self.discriminator += [
            torch.nn.Sequential(
                torch.nn.ReflectionPad2d((1, 1, 1, 1)),
                torch.nn.utils.weight_norm(
                    torch.nn.Conv2d(
                        in_channels,
                        hidden_channels // 32,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                    )
                ),
            ),
            torch.nn.Sequential(
                torch.nn.LeakyReLU(0.2, True),
                torch.nn.ReflectionPad2d((1, 1, 1, 1)),
                torch.nn.utils.weight_norm(
                    torch.nn.Conv2d(
                        hidden_channels // 32,
                        hidden_channels // 16,
                        kernel_size=(3, 3),
                        stride=(2, 2),
                    )
                ),
            ),
            torch.nn.Sequential(
                torch.nn.LeakyReLU(0.2, True),
                torch.nn.ReflectionPad2d((1, 1, 1, 1)),
                torch.nn.utils.weight_norm(
                    torch.nn.Conv2d(
                        hidden_channels // 16,
                        hidden_channels // 8,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                    )
                ),
            ),
            torch.nn.Sequential(
                torch.nn.LeakyReLU(0.2, True),
                torch.nn.ReflectionPad2d((1, 1, 1, 1)),
                torch.nn.utils.weight_norm(
                    torch.nn.Conv2d(
                        hidden_channels // 8,
                        hidden_channels // 4,
                        kernel_size=(3, 3),
                        stride=(2, 2),
                    )
                ),
            ),
            torch.nn.Sequential(
                torch.nn.LeakyReLU(0.2, True),
                torch.nn.ReflectionPad2d((1, 1, 1, 1)),
                torch.nn.utils.weight_norm(
                    torch.nn.Conv2d(
                        hidden_channels // 4,
                        hidden_channels // 2,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                    )
                ),
            ),
            torch.nn.Sequential(
                torch.nn.LeakyReLU(0.2, True),
                torch.nn.ReflectionPad2d((1, 1, 1, 1)),
                torch.nn.utils.weight_norm(
                    torch.nn.Conv2d(
                        hidden_channels // 2,
                        hidden_channels,
                        kernel_size=(3, 3),
                        stride=(2, 2),
                    )
                ),
            ),
            torch.nn.Sequential(
                torch.nn.LeakyReLU(0.2, True),
                torch.nn.ReflectionPad2d((1, 1, 1, 1)),
                torch.nn.utils.weight_norm(
                    torch.nn.Conv2d(
                        hidden_channels, 1, kernel_size=(3, 3), stride=(1, 1)
                    )
                ),
            ),
        ]

    def forward(self, x):
        hiddens = []
        for layer in self.discriminator:
            x = layer(x)
            hiddens.append(x)
        return x, hiddens[-1]


class VISinger2Discriminator(torch.nn.Module):
    def __init__(self, hps, use_spectral_norm=False):
        super().__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
        self.discriminators = torch.nn.ModuleList(discs)
        self.disc_multfrequency = MultiFrequencyDiscriminator(
            hop_lengths=[
                int(hps.data.sample_rate * 2.5 / 1000),
                int(hps.data.sample_rate * 5 / 1000),
                int(hps.data.sample_rate * 7.5 / 1000),
                int(hps.data.sample_rate * 10 / 1000),
                int(hps.data.sample_rate * 12.5 / 1000),
                int(hps.data.sample_rate * 15 / 1000),
            ],
            hidden_channels=[256, 256, 256, 256, 256],
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        scores_r, fmaps_r = self.disc_multfrequency(y)
        scores_g, fmaps_g = self.disc_multfrequency(y_hat)
        for i in range(len(scores_r)):
            y_d_rs.append(scores_r[i])
            y_d_gs.append(scores_g[i])
            fmap_rs.append(fmaps_r[i])
            fmap_gs.append(fmaps_g[i])
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


# TODO(Yifeng): Not sure if those modules exists in espnet.


class LayerNorm(torch.nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class ConvReluNorm(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        kernel_size,
        n_layers,
        p_dropout,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        assert n_layers > 1, "Number of layers should be larger than 0."

        self.conv_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        self.conv_layers.append(
            torch.nn.Conv1d(
                in_channels, hidden_channels, kernel_size, padding=kernel_size // 2
            )
        )
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = torch.nn.Sequential(
            torch.nn.ReLU(), torch.nn.Dropout(p_dropout)
        )
        for _ in range(n_layers - 1):
            self.conv_layers.append(
                torch.nn.Conv1d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                )
            )
            self.norm_layers.append(LayerNorm(hidden_channels))
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x):
        x = self.conv_layers[0](x)
        x = self.norm_layers[0](x)
        x = self.relu_drop(x)

        for i in range(1, self.n_layers):
            x_ = self.conv_layers[i](x)
            x_ = self.norm_layers[i](x_)
            x_ = self.relu_drop(x_)
            x = (x + x_) / 2
        x = self.proj(x)
        return x
