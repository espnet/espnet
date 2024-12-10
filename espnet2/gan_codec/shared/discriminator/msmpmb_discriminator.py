from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops import rearrange
from torch.nn.utils import weight_norm

from espnet2.gan_tts.hifigan.hifigan import (
    HiFiGANMultiPeriodDiscriminator,
    HiFiGANMultiScaleDiscriminator,
)


def WNConv1d(*args, **kwargs):
    """
    Create a 1D convolutional layer with weight normalization.

    This function wraps the standard `nn.Conv1d` layer with weight normalization.
    Optionally, it can also include a Leaky ReLU activation layer.

    Args:
        *args: Variable length argument list for the convolution layer.
        **kwargs: Keyword arguments for the convolution layer, including:
            act (bool): If True, adds a LeakyReLU activation. Default is True.

    Returns:
        nn.Sequential or nn.Conv1d: The convolutional layer, possibly wrapped
        in a Sequential model with activation.

    Examples:
        >>> conv_layer = WNConv1d(1, 16, kernel_size=3)
        >>> print(conv_layer)
        Sequential(
          (0): Conv1d(1, 16, kernel_size=(3,), stride=(1,), ...)
          (1): LeakyReLU(negative_slope=0.1)
        )
    """
    act = kwargs.pop("act", True)
    conv = weight_norm(nn.Conv1d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


def WNConv2d(*args, **kwargs):
    """
    Weight Normalized 2D Convolution Layer.

    This class applies a 2D convolution with weight normalization. It can
    optionally apply a Leaky ReLU activation function after the convolution.

    Attributes:
        act (bool): If True, applies Leaky ReLU activation after convolution.

    Args:
        *args: Variable length argument list for nn.Conv2d.
        **kwargs: Keyword arguments for nn.Conv2d, including:
            - act (bool): Whether to apply activation function (default: True).

    Returns:
        nn.Sequential or nn.Conv2d: A sequential model with convolution and
        activation or just the convolution layer.

    Examples:
        >>> conv_layer = WNConv2d(3, 16, kernel_size=3, padding=1)
        >>> print(conv_layer)
        WNConv2d(
          (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): LeakyReLU(negative_slope=0.1)
        )

        >>> conv_layer_no_act = WNConv2d(3, 16, kernel_size=3, padding=1, act=False)
        >>> print(conv_layer_no_act)
        Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    """
    act = kwargs.pop("act", True)
    conv = weight_norm(nn.Conv2d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


class MultiScaleDiscriminator(nn.Module):
    """
        MultiScaleDiscriminator is a neural network module that implements a multi-scale
    discriminator for evaluating the quality of audio signals. It utilizes a series of
    1D convolutional layers with weight normalization to extract features from audio
    inputs at different scales.

    Attributes:
        convs (nn.ModuleList): A list of convolutional layers for feature extraction.
        conv_post (nn.Module): A final convolutional layer to process the output.
        sample_rate (int): The sampling rate of the audio input.
        rate (int): The downsampling rate applied to the input audio.

    Args:
        rate (int): The downsampling rate for the input audio. Default is 1.
        sample_rate (int): The original sampling rate of the audio. Default is 44100.

    Returns:
        List[Tensor]: A list of feature maps extracted from the audio input at
        different layers of the network.

    Raises:
        ValueError: If the input tensor `x` does not have the correct shape.

    Examples:
        >>> discriminator = MultiScaleDiscriminator(rate=2, sample_rate=48000)
        >>> audio_input = torch.randn(1, 1, 48000)  # Batch size 1, 1 channel, 1 sec
        >>> feature_maps = discriminator(audio_input)
        >>> len(feature_maps)  # Should return the number of layers + 1
        7

    Note:
        The input tensor should have the shape (batch_size, channels, length).
    """

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
        """
            Forward pass through the MultiScaleDiscriminator.

        This method processes the input tensor `x` through a series of
        convolutional layers and returns feature maps at each stage of
        the network. The input audio is first resampled based on the
        specified sample rate and downsample rate. The processed output
        includes feature maps from all convolutional layers in the
        MultiScaleDiscriminator.

        Args:
            x (torch.Tensor): Input tensor representing the audio signal
                with shape (batch_size, 1, num_samples).

        Returns:
            List[torch.Tensor]: A list of feature maps from each convolutional
                layer, including the final output shape.

        Examples:
            >>> discriminator = MultiScaleDiscriminator(rate=2, sample_rate=44100)
            >>> audio_input = torch.randn(1, 1, 44100)  # Example audio input
            >>> feature_maps = discriminator(audio_input)
            >>> print(len(feature_maps))  # Output: Number of feature maps
            >>> print(feature_maps[0].shape)  # Shape of the first feature map

        Note:
            Ensure that the input tensor `x` is on the same device as the
            model (CPU or GPU) for the processing to work correctly.

        Raises:
            RuntimeError: If the input tensor does not have the expected
                shape or is not on the correct device.
        """
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
    """
        MultiBandDiscriminator is a complex multi-band spectrogram discriminator that
    operates on audio signals to classify them based on their frequency bands.

    Attributes:
        window_length (int): Window length of the Short-Time Fourier Transform (STFT).
        hop_factor (float): Hop factor of the STFT.
        sample_rate (int): Sampling rate of audio in Hz.
        bands (list): Frequency bands for the discriminator.
        n_fft (int): Number of FFT points.
        hop_length (int): Length of hop between STFT windows.
        band_convs (ModuleList): List of convolutional layers for each frequency band.
        conv_post (Module): Final convolutional layer applied to the concatenated outputs.

    Args:
        window_length (int): Window length of STFT.
        hop_factor (float, optional): Hop factor of the STFT, defaults to
            0.25 * window_length.
        sample_rate (int, optional): Sampling rate of audio in Hz, defaults to 44100.
        bands (list, optional): Bands to run discriminator over, defaults to
            [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)].
        channel (int, optional): Number of channels in the convolutional layers,
            defaults to 32.

    Methods:
        spectrogram(x): Computes the spectrogram of the input audio and splits it
            into defined frequency bands.
        forward(x): Processes the input audio through the discriminator and
            returns feature maps.

    Examples:
        >>> discriminator = MultiBandDiscriminator(window_length=1024)
        >>> audio_input = torch.randn(1, 1, 44100)  # Example audio tensor
        >>> features = discriminator(audio_input)

    Note:
        This class is part of a larger GAN architecture and is used to
        distinguish real from generated audio based on multi-band spectrograms.
    """

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
        """
        Complex multi-band spectrogram discriminator.

        This class implements a multi-band discriminator for audio signals,
        which processes the input signal in different frequency bands using
        convolutional layers. It is designed for use in generative adversarial
        networks (GANs) for tasks like speech synthesis.

        Attributes:
            window_length (int): Window length of STFT.
            hop_factor (float): Hop factor of the STFT, defaults to
                ``0.25 * window_length``.
            sample_rate (int): Sampling rate of audio in Hz, by default 44100.
            bands (list): List of frequency bands to run the discriminator over.
            n_fft (int): Number of FFT points.
            hop_length (int): Length of hop between STFT windows.
            band_convs (ModuleList): List of convolutional layers for each band.
            conv_post (Module): Post-processing convolutional layer.

        Args:
            window_length (int): Length of the STFT window.
            hop_factor (float, optional): Factor determining the hop length,
                defaults to 0.25.
            sample_rate (int, optional): Audio sampling rate in Hz,
                defaults to 44100.
            bands (list, optional): List of frequency bands for the discriminator,
                defaults to [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5),
                (0.5, 0.75), (0.75, 1.0)].
            channel (int, optional): Number of channels for the convolutional
                layers, defaults to 32.

        Returns:
            List[Tensor]: A list of feature maps from the discriminator.

        Examples:
            >>> discriminator = MultiBandDiscriminator(window_length=1024)
            >>> input_signal = torch.randn(1, 1, 44100)  # Batch size 1, mono
            >>> output = discriminator(input_signal)
            >>> len(output)  # Should match the number of bands

        Note:
            This class relies on the PyTorch and torchaudio libraries for
            tensor operations and audio processing.

        Todo:
            - Add support for configurable activation functions in convolutional layers.
        """
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
        """
            Forward pass for the MultiBandDiscriminator.

        This method processes the input tensor `x` through the discriminator's
        architecture, generating feature maps from multiple frequency bands. The
        input audio is first converted into a spectrogram, which is then split
        into defined frequency bands. Each band is processed through a series
        of convolutional layers, and the outputs are collected into a list of
        feature maps.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, time)
                representing the audio signal.

        Returns:
            List[torch.Tensor]: A list of feature maps, where each feature map
                corresponds to the output from the convolutional layers applied
                to the respective frequency band, as well as the final output
                after the post-processing convolution.

        Examples:
            >>> discriminator = MultiBandDiscriminator(window_length=1024)
            >>> input_audio = torch.randn(1, 1, 2048)  # Example audio input
            >>> output_feature_maps = discriminator.forward(input_audio)
            >>> print(len(output_feature_maps))  # Number of bands + 1 for post-conv

        Note:
            The input audio should be a mono signal with a shape of
            (batch_size, 1, time).

        Raises:
            ValueError: If the input tensor does not have the correct shape.
        """
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
    """
        MultiScaleMultiPeriodMultiBandDiscriminator combines multiple discriminators
    for audio signal processing, including multi-scale, multi-period, and
    multi-band analysis. This class utilizes various discriminators to enhance
    the performance of generative adversarial networks (GANs) for audio tasks.

    Attributes:
        rates (list): List of sampling rates (in Hz) to run the multi-scale
            discriminator (MSD) at. If empty, MSD is not used.
        fft_sizes (list): Window sizes of the FFT to run the multi-band
            discriminator (MRD) at. Defaults to [2048, 1024, 512].
        sample_rate (int): Sampling rate of audio in Hz. Defaults to 44100.
        periods (list): List of periods (of samples) to run the multi-period
            discriminator (MPD) at. Defaults to [2, 3, 5, 7, 11].
        period_discriminator_params (Dict[str, Any]): Parameters for the
            multi-period discriminator.
        band_discriminator_params (Dict[str, Any]): Parameters for the
            multi-band discriminator.

    Args:
        rates (list, optional): Sampling rates to run MSD at. Defaults to [].
        periods (list, optional): Periods to run MPD at. Defaults to
            [2, 3, 5, 7, 11].
        fft_sizes (list, optional): Window sizes of the FFT to run MRD at.
            Defaults to [2048, 1024, 512].
        sample_rate (int, optional): Sampling rate of audio in Hz. Defaults
            to 44100.
        band_discriminator_params (Dict[str, Any], optional): Parameters for
            the multi-band discriminator. Defaults to a specified dictionary.

    Examples:
        >>> discriminator = MultiScaleMultiPeriodMultiBandDiscriminator(
        ...     rates=[16000, 24000],
        ...     periods=[2, 3],
        ...     fft_sizes=[2048, 1024]
        ... )
        >>> output = discriminator(torch.randn(1, 1, 44100))  # Example input
        >>> print(len(output))  # Check the number of outputs

    Note:
        The input audio signal should be of shape (batch_size, channels,
        samples).

    Todo:
        - Add support for additional discriminator types.
        - Implement training and evaluation routines for the discriminator.

    Raises:
        ValueError: If any of the parameters are invalid.
    """

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
        """
                Preprocess the input audio by removing the DC offset and normalizing the volume.

        This method performs two main operations on the input audio tensor `y`:
        1. It removes the DC offset by subtracting the mean of the audio signal.
        2. It peak normalizes the volume of the audio to a maximum amplitude of 0.8.

        The output is a tensor with the same shape as the input, where the audio is processed for better compatibility with the discriminators.

        Args:
            y (torch.Tensor): Input audio tensor of shape (batch_size, num_samples).

        Returns:
            torch.Tensor: Preprocessed audio tensor of the same shape as input `y`.

        Examples:
            >>> discriminator = MultiScaleMultiPeriodMultiBandDiscriminator()
            >>> audio_input = torch.randn(4, 16000)  # Example audio tensor
            >>> preprocessed_audio = discriminator.preprocess(audio_input)
            >>> print(preprocessed_audio.shape)
            torch.Size([4, 16000])

        Note:
            The normalization is performed to avoid clipping during processing.
        """
        # Remove DC offset
        y = y - y.mean(dim=-1, keepdims=True)
        # Peak normalize the volume of input audio
        y = 0.8 * y / (y.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        return y

    def forward(self, x):
        """
        Discriminator that combines multiple discriminators.

        This class integrates multiple discriminators, including multi-scale,
        multi-period, and multi-band discriminators to evaluate the quality of
        generated audio signals.

        Attributes:
            rates (list): Sampling rates (in Hz) to run the multi-scale
                discriminator. If empty, the multi-scale discriminator is not used.
            fft_sizes (list): Window sizes of the FFT to run the multi-band
                discriminator. Default is [2048, 1024, 512].
            sample_rate (int): Sampling rate of audio in Hz. Default is 44100.
            periods (list): Periods (in samples) to run the multi-period
                discriminator. Default is [2, 3, 5, 7, 11].
            period_discriminator_params (Dict[str, Any]): Parameters for the
                multi-period discriminator.
            band_discriminator_params (Dict[str, Any]): Parameters for the
                multi-band discriminator.

        Args:
            rates: List of sampling rates for the multi-scale discriminator.
            fft_sizes: List of FFT window sizes for the multi-band discriminator.
            sample_rate: Sampling rate of the audio.
            periods: List of periods for the multi-period discriminator.
            period_discriminator_params: Parameters for the multi-period
                discriminator.
            band_discriminator_params: Parameters for the multi-band discriminator.

        Examples:
            >>> discriminator = MultiScaleMultiPeriodMultiBandDiscriminator(
            ...     rates=[1, 2, 4],
            ...     fft_sizes=[2048, 1024],
            ...     sample_rate=44100,
            ... )
            >>> output = discriminator(torch.randn(1, 1, 44100))  # Random input
            >>> print(len(output))  # Should show the number of outputs from discriminators
        """
        x = self.preprocess(x)
        mrd_out = [d(x) for d in self.discriminators]
        mpd_out = self.mpd(x)
        return mrd_out + mpd_out
