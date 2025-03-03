from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
import torchaudio
from einops import rearrange
from torch.nn.utils import weight_norm

from espnet2.gan_tts.hifigan.hifigan import (  # noqa
    HiFiGANMultiPeriodDiscriminator,
    HiFiGANMultiScaleDiscriminator,
)


def WNConv1d(*args, **kwargs):
    act = kwargs.pop("act", True)
    conv = weight_norm(nn.Conv1d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


def WNConv2d(*args, **kwargs):
    act = kwargs.pop("act", True)
    conv = weight_norm(nn.Conv2d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, rate: int = 1, sample_rate: int = 44100):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                WNConv1d(1, 16, 15, 1, padding=7),
                WNConv1d(16, 64, 41, 4, groups=4, padding=20),
                WNConv1d(64, 256, 41, 4, groups=16, padding=20),
                WNConv1d(256, 1024, 41, 4, groups=64, padding=20),
                WNConv1d(1024, 1024, 41, 4, groups=256, padding=20),
                WNConv1d(1024, 1024, 5, 1, padding=2),
            ]
        )
        self.conv_post = WNConv1d(1024, 1, 3, 1, padding=1, act=False)
        self.sample_rate = sample_rate
        self.rate = rate

    def forward(self, x):
        resample_transform = torchaudio.transforms.Resample(
            orig_freq=self.sample_rate, new_freq=self.sample_rate // self.rate
        ).to(x.device)
        x = resample_transform(x)

        fmap = []

        for layer in self.convs:
            x = layer(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)

        return fmap


# BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]


class MultiBandDiscriminator(nn.Module):
    def __init__(
        self,
        window_length: int,
        hop_factor: float = 0.25,
        sample_rate: int = 44100,
        bands: list = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)],
        channel: int = 32,
    ):
        """Complex multi-band spectrogram discriminator.

        Parameters
        ----------
        window_length : int
            Window length of STFT.
        hop_factor : float, optional
            Hop factor of the STFT, defaults to ``0.25 * window_length``.
        sample_rate : int, optional
            Sampling rate of audio in Hz, by default 44100
        bands : list, optional
            Bands to run discriminator over.
        """
        super().__init__()

        self.window_length = window_length
        self.hop_factor = hop_factor
        self.sample_rate = sample_rate

        n_fft = window_length // 2 + 1
        bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.bands = bands
        self.n_fft = n_fft
        self.hop_length = int(window_length * hop_factor)

        def convs():
            return nn.ModuleList(
                [
                    WNConv2d(2, channel, (3, 9), (1, 1), padding=(1, 4)),
                    WNConv2d(channel, channel, (3, 9), (1, 2), padding=(1, 4)),
                    WNConv2d(channel, channel, (3, 9), (1, 2), padding=(1, 4)),
                    WNConv2d(channel, channel, (3, 9), (1, 2), padding=(1, 4)),
                    WNConv2d(channel, channel, (3, 3), (1, 1), padding=(1, 1)),
                ]
            )

        self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])
        self.conv_post = WNConv2d(channel, 1, (3, 3), (1, 1), padding=(1, 1), act=False)

    def spectrogram(self, x):
        stft = torchaudio.transforms.Spectrogram(
            n_fft=self.window_length,
            win_length=self.window_length,
            hop_length=self.hop_length,
            power=None,
            return_complex=True,
        ).to(x.device)
        x = stft(x)
        x = torch.view_as_real(x)
        x = rearrange(x, "b 1 f t c -> (b 1) c t f")
        # Split into bands
        x_bands = [x[..., b[0] : b[1]] for b in self.bands]
        return x_bands

    def forward(self, x):
        x_bands = self.spectrogram(x)
        fmap = []

        x = []
        for band, stack in zip(x_bands, self.band_convs):
            for layer in stack:
                band = layer(band)
                fmap.append(band)
            x.append(band)

        x = torch.cat(x, dim=-1)
        x = self.conv_post(x)
        fmap.append(x)

        return fmap


class MultiScaleMultiPeriodMultiBandDiscriminator(nn.Module):
    def __init__(
        self,
        rates: list = [],
        fft_sizes: list = [2048, 1024, 512],
        sample_rate: int = 44100,
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
        band_discriminator_params: Dict[str, Any] = {
            "hop_factor": 0.25,
            "sample_rate": 24000,
            "bands": [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)],
            "channel": 32,
        },
    ):
        """Discriminator that combines multiple discriminators.

        Parameters
        ----------
        rates : list, optional
            sampling rates (in Hz) to run MSD at, by default []
            If empty, MSD is not used.
        periods : list, optional
            periods (of samples) to run MPD at, by default [2, 3, 5, 7, 11]
        fft_sizes : list, optional
            Window sizes of the FFT to run MRD at, by default [2048, 1024, 512]
        sample_rate : int, optional
            Sampling rate of audio in Hz, by default 44100
        bands : list, optional
            Bands to run MRD at, by default `BANDS`
        """
        super().__init__()
        discs = []
        self.mpd = HiFiGANMultiPeriodDiscriminator(
            periods=periods,
            discriminator_params=period_discriminator_params,
        )

        # discs += [MPD(p) for p in periods]
        discs += [MultiScaleDiscriminator(r, sample_rate=sample_rate) for r in rates]
        discs += [
            MultiBandDiscriminator(
                f,
                sample_rate=band_discriminator_params["sample_rate"],
                bands=band_discriminator_params["bands"],
            )
            for f in fft_sizes
        ]
        self.discriminators = nn.ModuleList(discs)

    def preprocess(self, y):
        # Remove DC offset
        y = y - y.mean(dim=-1, keepdims=True)
        # Peak normalize the volume of input audio
        y = 0.8 * y / (y.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        return y

    def forward(self, x):
        x = self.preprocess(x)
        mrd_out = [d(x) for d in self.discriminators]
        mpd_out = self.mpd(x)
        return mrd_out + mpd_out
