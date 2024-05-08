# Copyright 2023 Yifeng Yu
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""VISinger2 HiFi-GAN Modules.

This code is based on https://github.com/zhangyongmao/VISinger2

"""

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.gan_svs.visinger2.ddsp import (
    remove_above_nyquist,
    scale_function,
    upsample,
)
from espnet2.gan_tts.hifigan import (
    HiFiGANMultiPeriodDiscriminator,
    HiFiGANMultiScaleDiscriminator,
)
from espnet2.gan_tts.hifigan.residual_block import ResidualBlock


class VISinger2VocoderGenerator(torch.nn.Module):

    @typechecked
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
            n_harmonic (int): Number of harmonics used to synthesize a sound signal.
            use_additional_convs (bool): Whether to use additional conv layers in
                residual blocks.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (Dict[str, Any]): Hyperparameters for activation
                function.
            use_weight_norm (bool): Whether to use weight norm. If set to true, it will
                be applied to all of the conv layers.

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
            ddsp (Tensor): Input tensor (B, n_harmonic + 2, T * hop_length).
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
                cs = cs + self.blocks[i * self.num_blocks + j](c)
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
        dropout_rate: float = 0.1,
        sample_rate: int = 22050,
        hop_size: int = 256,
    ):
        """Initialize harmonic generator module.

        Args:
            hidden_channels (int): Number of channels in the input and hidden layers.
            n_harmonic (int): Number of harmonic channels.
            kernel_size (int): Size of the convolutional kernel.
            padding (int): Amount of padding added to the input.
            dropout_rate (float): Dropout rate.
            sample_rate (int): Sampling rate of the input audio.
            hop_size (int): Hop size used in the analysis of the input audio.

        """
        super().__init__()

        self.prenet = torch.nn.Conv1d(
            hidden_channels, hidden_channels, kernel_size, padding=padding
        )

        self.net = ConvReluNorm(
            hidden_channels,
            hidden_channels,
            hidden_channels,
            kernel_size,
            8,
            dropout_rate,
        )

        self.postnet = torch.nn.Conv1d(
            hidden_channels, n_harmonic + 1, kernel_size, padding=padding
        )

        self.sample_rate = sample_rate
        self.hop_size = hop_size

    def forward(self, f0, harm, mask):
        """Generate harmonics from F0 and harmonic data.

        Args:
            f0 (Tensor): Pitch (F0) tensor (B, 1, T).
            harm (Tensor): Harmonic data tensor (B, hidden_channels, T).
            mask (Tensor): Mask tensor for harmonic data (B, 1, T).

        Returns:
            Tensor: Harmonic signal tensor (B, n_harmonic, T * hop_length).

        """

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
        dropout_rate: float = 0.1,
    ):
        """Initialize the Generator_Noise module.

        Args:
            win_length (int, optional): Window length. If None, set to `n_fft`.
            hop_length (int): Hop length.
            n_fft (int): FFT size.
            hidden_channels (int): Number of hidden representation channels.
            kernel_size (int): Size of the convolutional kernel.
            padding (int): Size of the padding applied to the input.
            dropout_rate (float): Dropout rate.
        """

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
            dropout_rate,
        )

        self.istft_amplitude = torch.nn.Conv1d(
            hidden_channels, self.fft_size // 2 + 1, 1, padding
        )
        self.window = torch.hann_window(self.win_size)

    def forward(self, x, mask):
        """Forward Generator Noise.

        Args:
            x (Tensor): Input tensor (B, hidden_channels, T).
            mask (Tensor): Mask tensor (B, 1, T).
        Returns:
            Tensor: Output tensor (B, 1, T * hop_size).
        """
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
    """Multi-Frequency Discriminator module in UnivNet."""

    def __init__(
        self,
        sample_rate: int = 22050,
        hop_lengths=[128, 256, 512],
        hidden_channels=[256, 512, 512],
        domain="double",
        mel_scale=True,
        divisors=[32, 16, 8, 4, 2, 1, 1],
        strides=[1, 2, 1, 2, 1, 2, 1],
    ):
        """Initialize Multi-Frequency Discriminator module.

        Args:
            hop_lengths (list): List of hop lengths.
            hidden_channels (list): List of number of channels in hidden layers.
            domain (str): Domain of input signal. Default is "double".
            mel_scale (bool): Whether to use mel-scale frequency. Default is True.
            divisors (list): List of divisors for each layer in the discriminator.
                             Default is [32, 16, 8, 4, 2, 1, 1].
            strides (list): List of strides for each layer in the discriminator.
                            Default is [1, 2, 1, 2, 1, 2, 1].
        """

        super().__init__()

        # TODO(Yifeng): Maybe use LogMelFbank instead of TorchSTFT
        self.stfts = torch.nn.ModuleList(
            [
                TorchSTFT(
                    sample_rate=sample_rate,
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
                    BaseFrequenceDiscriminator(2, c, divisors=divisors, strides=strides)
                    for x, c in zip(hop_lengths, hidden_channels)
                ]
            )
        else:
            self.discriminators = torch.nn.ModuleList(
                [
                    BaseFrequenceDiscriminator(1, c, divisors=divisors, strides=strides)
                    for x, c in zip(hop_lengths, hidden_channels)
                ]
            )

    def forward(self, x):
        """Forward pass of Multi-Frequency Discriminator module.

        Args:
            x (Tensor): Input tensor (B, 1, T * hop_size).

        Returns:
            List[Tensor]: List of feature maps.
        """

        feats = list()
        for stft, layer in zip(self.stfts, self.discriminators):
            mag, phase = stft.transform(x.squeeze(1))
            if self.domain == "double":
                mag = torch.stack(torch.chunk(mag, 2, dim=1), dim=1)
            else:
                mag = mag.unsqueeze(1)

            feat = layer(mag)
            feats.append(feat)
        return feats


class BaseFrequenceDiscriminator(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=512,
        divisors=[32, 16, 8, 4, 2, 1, 1],
        strides=[1, 2, 1, 2, 1, 2, 1],
    ):
        """Base Frequence Discriminator

        Args:
            in_channels (int): Number of input channels.
            hidden_channels (int, optional): Number of channels in hidden layers.
                                             Defaults to 512.
            divisors (List[int], optional): List of divisors for the number of channels
                                            in each layer. The length of the list
                                            determines the number of layers. Defaults
                                            to [32, 16, 8, 4, 2, 1, 1].
            strides (List[int], optional): List of stride values for each layer. The
                                           length of the list determines the number
                                           of layers.Defaults to [1, 2, 1, 2, 1, 2, 1].
        """

        super().__init__()

        layers = []
        for i in range(len(divisors) - 1):
            in_ch = in_channels if i == 0 else hidden_channels // divisors[i - 1]
            out_ch = hidden_channels // divisors[i]
            stride = strides[i]
            layers.append((in_ch, out_ch, stride))
        layers.append((hidden_channels // divisors[-1], 1, strides[-1]))
        self.discriminators = torch.nn.ModuleList()

        for in_ch, out_ch, stride in layers:
            seq = torch.nn.Sequential(
                torch.nn.LeakyReLU(0.2, True) if out_ch != 1 else torch.nn.Identity(),
                torch.nn.ReflectionPad2d((1, 1, 1, 1)),
                torch.nn.utils.weight_norm(
                    torch.nn.Conv2d(
                        in_ch,
                        out_ch,
                        kernel_size=(3, 3),
                        stride=(stride, stride),
                    )
                ),
            )
            self.discriminators += [seq]

    def forward(self, x):
        """Perform forward pass through the base frequency discriminator.

        Args:
            x (torch.Tensor): Input tensor of shape
                              (B, in_channels, freq_bins, time_steps).

        Returns:
            List[torch.Tensor]: List of output tensors from each layer of the
                                discriminator, where the first tensor corresponds to
                                the output of the first layer, and so on.
        """

        outs = []
        for f in self.discriminators:
            x = f(x)
            outs = outs + [x]

        return outs


class VISinger2Discriminator(torch.nn.Module):
    def __init__(
        self,
        # Multi-scale discriminator related
        scales: int = 1,
        scale_downsample_pooling: str = "AvgPool1d",
        scale_downsample_pooling_params: Dict[str, Any] = {
            "kernel_size": 4,
            "stride": 2,
            "padding": 2,
        },
        scale_discriminator_params: Dict[str, Any] = {
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [15, 41, 5, 3],
            "channels": 128,
            "max_downsample_channels": 1024,
            "max_groups": 16,
            "bias": True,
            "downsample_scales": [2, 2, 4, 4, 1],
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
        },
        follow_official_norm: bool = True,
        # Multi-period discriminator related
        periods: List[int] = [2, 3, 5, 7, 11],
        period_discriminator_params: Dict[str, Any] = {
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [5, 3],
            "channels": 32,
            "downsample_scales": [3, 3, 3, 3, 1],
            "max_downsample_channels": 1024,
            "bias": True,
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
            "use_weight_norm": True,
            "use_spectral_norm": False,
        },
        # Multi-frequency discriminator related
        multi_freq_disc_params: Dict[str, Any] = {
            "sample_rate": 22050,
            "hop_length_factors": [4, 8, 16],
            "hidden_channels": [256, 512, 512],
            "domain": "double",
            "mel_scale": True,
            "divisors": [32, 16, 8, 4, 2, 1, 1],
            "strides": [1, 2, 1, 2, 1, 2, 1],
        },
    ):
        """Discriminator module for VISinger2, including MSD, MPD, and MFD.

        Args:
            scales (int): Number of scales to be used in the multi-scale discriminator.
            scale_downsample_pooling (str): Type of pooling used for downsampling.
            scale_downsample_pooling_params (Dict[str, Any]): Parameters for the
                                                              downsampling pooling
                                                              layer.
            scale_discriminator_params (Dict[str, Any]): Parameters for the scale
                                                         discriminator.
            follow_official_norm (bool): Whether to follow the official normalization.
            periods (List[int]): List of periods to be used in the multi-period
                                 discriminator.
            period_discriminator_params (Dict[str, Any]): Parameters for the period
                                                          discriminator.
            multi_freq_disc_params (Dict[str, Any]): Parameters for the
                                                     multi-frequency discriminator.
            use_spectral_norm (bool): Whether to use spectral normalization or not.
        """
        super().__init__()

        # Multi-scale discriminator related
        self.msd = HiFiGANMultiScaleDiscriminator(
            scales=scales,
            downsample_pooling=scale_downsample_pooling,
            downsample_pooling_params=scale_downsample_pooling_params,
            discriminator_params=scale_discriminator_params,
            follow_official_norm=follow_official_norm,
        )

        # Multi-period discriminator related
        self.mpd = HiFiGANMultiPeriodDiscriminator(
            periods=periods,
            discriminator_params=period_discriminator_params,
        )

        # Multi-frequency discriminator related
        if "hop_lengths" not in multi_freq_disc_params:
            # Transfer hop lengths factors to hop lengths
            multi_freq_disc_params["hop_lengths"] = []

            for i in range(len(multi_freq_disc_params["hop_length_factors"])):
                multi_freq_disc_params["hop_lengths"].append(
                    int(
                        multi_freq_disc_params["sample_rate"]
                        * multi_freq_disc_params["hop_length_factors"][i]
                        / 1000
                    )
                )

            del multi_freq_disc_params["hop_length_factors"]

        self.mfd = MultiFrequencyDiscriminator(
            **multi_freq_disc_params,
        )

    def forward(self, x):
        msd_outs = self.msd(x)
        mpd_outs = self.mpd(x)
        mfd_outs = self.mfd(x)
        return msd_outs + mpd_outs + mfd_outs


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
        dropout_rate,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
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
            torch.nn.ReLU(), torch.nn.Dropout(dropout_rate)
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


class TorchSTFT(torch.nn.Module):
    def __init__(
        self,
        sample_rate,
        fft_size,
        hop_size,
        win_size,
        normalized=False,
        domain="linear",
        mel_scale=False,
        ref_level_db=20,
        min_level_db=-100,
    ):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.ref_level_db = ref_level_db
        self.min_level_db = min_level_db
        self.window = torch.hann_window(win_size)
        self.normalized = normalized
        self.domain = domain
        self.mel_scale = (
            MelScale(
                sample_rate=sample_rate,
                n_mels=(fft_size // 2 + 1),
                n_stft=(fft_size // 2 + 1),
            )
            if mel_scale
            else None
        )

    def transform(self, x):
        x_stft = torch.stft(
            x,
            self.fft_size,
            self.hop_size,
            self.win_size,
            self.window.type_as(x),
            normalized=self.normalized,
            return_complex=True,
        )

        real = torch.real(x_stft)
        imag = torch.imag(x_stft)
        mag = torch.clamp(real**2 + imag**2, min=1e-7)
        mag = torch.sqrt(mag)
        phase = torch.angle(x_stft)

        if self.mel_scale is not None:
            mag = self.mel_scale(mag)

        if self.domain == "log":
            mag = 20 * torch.log10(mag) - self.ref_level_db
            mag = torch.clamp((mag - self.min_level_db) / -self.min_level_db, 0, 1)
            return mag, phase
        elif self.domain == "linear":
            return mag, phase
        elif self.domain == "double":
            log_mag = 20 * torch.log10(mag) - self.ref_level_db
            log_mag = torch.clamp(
                (log_mag - self.min_level_db) / -self.min_level_db, 0, 1
            )
            return torch.cat((mag, log_mag), dim=1), phase

    def complex(self, x):
        x_stft = torch.stft(
            x,
            self.fft_size,
            self.hop_size,
            self.win_size,
            self.window.type_as(x),
            normalized=self.normalized,
        )
        real = x_stft[..., 0]
        imag = x_stft[..., 1]
        return real, imag


class MelScale(torch.nn.Module):
    """Turn a normal STFT into a mel frequency STFT, using a conversion

    matrix.  This uses triangular filter banks.
    User can control which device the filter bank (fb) is (e.g. fb.to(spec_f.device)).
    Args:
        n_mels (int, optional): Number of mel filterbanks. (Default: 128)
        sample_rate (int, optional): Sample rate of audio signal. (Default: 16000)
        f_min (float, optional): Minimum frequency. (Default: 0.)
        f_max (float or None, optional): Maximum frequency.
            (Default: sample_rate // 2)
        n_stft (int, optional): Number of bins in STFT. Calculated from first input
            if None is given.  See n_fft in :class:Spectrogram.
            (Default: None)
    """

    __constants__ = ["n_mels", "sample_rate", "f_min", "f_max"]

    def __init__(
        self,
        n_mels: int = 128,
        sample_rate: int = 24000,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        n_stft: Optional[int] = None,
    ) -> None:
        super(MelScale, self).__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.f_min = f_min

        assert f_min <= self.f_max, "Require f_min: %f < f_max: %f" % (
            f_min,
            self.f_max,
        )

        fb = (
            torch.empty(0)
            if n_stft is None
            else create_fb_matrix(
                n_stft, self.f_min, self.f_max, self.n_mels, self.sample_rate
            )
        )
        self.register_buffer("fb", fb)

    def forward(self, specgram: torch.Tensor) -> torch.Tensor:
        """Forward MelScale

        Args:
            specgram (Tensor): A spectrogram STFT of dimension (..., freq, time).
        Returns:
            Tensor: Mel frequency spectrogram of size (..., n_mels, time).
        """

        # pack batch
        shape = specgram.size()
        specgram = specgram.reshape(-1, shape[-2], shape[-1])

        if self.fb.numel() == 0:
            tmp_fb = create_fb_matrix(
                specgram.size(1), self.f_min, self.f_max, self.n_mels, self.sample_rate
            )
            # Attributes cannot be reassigned outside __init__ so workaround
            self.fb.resize_(tmp_fb.size())
            self.fb.copy_(tmp_fb)

        # (channel, frequency, time).transpose(...) dot (frequency, n_mels)
        # -> (channel, time, n_mels).transpose(...)
        mel_specgram = torch.matmul(specgram.transpose(1, 2), self.fb).transpose(1, 2)

        # unpack batch
        mel_specgram = mel_specgram.reshape(shape[:-2] + mel_specgram.shape[-2:])

        return mel_specgram


def create_fb_matrix(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_mels: int,
    sample_rate: int,
    norm: Optional[str] = None,
) -> torch.Tensor:
    """Create a frequency bin conversion matrix.

    Args:
        n_freqs (int): Number of frequencies to highlight/apply
        f_min (float): Minimum frequency (Hz)
        f_max (float): Maximum frequency (Hz)
        n_mels (int): Number of mel filterbanks
        sample_rate (int): Sample rate of the audio waveform
        norm (Optional[str]): If 'slaney',
        divide the triangular mel weights by the width of the mel band
        (area normalization). (Default: None)
    Returns:
        Tensor: Triangular filter banks (fb matrix) of size (n_freqs, n_mels)
        meaning number of frequencies to highlight/apply to x the number of filterbanks.
        Each column is a filterbank so that assuming there is a matrix A of
        size (..., n_freqs), the applied result would be
        A * create_fb_matrix(A.size(-1), ...).
    """

    if norm is not None and norm != "slaney":
        raise ValueError("norm must be one of None or 'slaney'")

    # freq bins
    # Equivalent filterbank construction by Librosa
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    # calculate mel freq bins
    # hertz to mel(f) is 2595. * math.log10(1. + (f / 700.))
    m_min = 2595.0 * math.log10(1.0 + (f_min / 700.0))
    m_max = 2595.0 * math.log10(1.0 + (f_max / 700.0))
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    # mel to hertz(mel) is 700. * (10**(mel / 2595.) - 1.)
    f_pts = 700.0 * (10 ** (m_pts / 2595.0) - 1.0)
    # calculate the difference between each mel point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_mels + 1)
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_mels + 2)
    # create overlapping triangles
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_mels)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_mels)
    fb = torch.min(down_slopes, up_slopes)
    fb = torch.clamp(fb, 1e-6, 1)

    if norm is not None and norm == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
        fb *= enorm.unsqueeze(0)
    return fb
