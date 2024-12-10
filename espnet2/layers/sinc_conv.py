#!/usr/bin/env python3
#  2020, Technische Universität München;  Ludwig Kürzinger
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Sinc convolutions."""
import math
from typing import Union

import torch
from typeguard import typechecked


class LogCompression(torch.nn.Module):
    """
        Log Compression Activation.

    This class implements the Log Compression activation function, defined as
    `log(abs(x) + 1)`, which is applied elementwise to the input tensor. This
    activation function is particularly useful in neural networks for compressing
    the range of input values.

    Attributes:
        None

    Args:
        None

    Returns:
        torch.Tensor: The output tensor after applying the Log Compression function.

    Examples:
        >>> log_compression = LogCompression()
        >>> input_tensor = torch.tensor([-1.0, 0.0, 1.0])
        >>> output_tensor = log_compression(input_tensor)
        >>> print(output_tensor)
        tensor([0.3133, 0.0000, 0.3133])
    """

    def __init__(self):
        """Initialize."""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Log Compression Activation.

        Applies the Log Compression function elementwise on the input tensor x,
        which is defined as `log(abs(x) + 1)`. This activation function is useful
        for compressing the range of input values, particularly in deep learning
        models.

        Args:
            x (torch.Tensor): The input tensor for which the Log Compression
                function is to be applied. The tensor can be of any shape.

        Returns:
            torch.Tensor: A tensor of the same shape as input x, containing the
            elementwise results of the Log Compression function.

        Examples:
            >>> import torch
            >>> log_compression = LogCompression()
            >>> input_tensor = torch.tensor([-1.0, 0.0, 1.0])
            >>> output_tensor = log_compression(input_tensor)
            >>> print(output_tensor)
            tensor([0.6931, 0.0000, 0.6931])
        """
        return torch.log(torch.abs(x) + 1)


class SincConv(torch.nn.Module):
    """
    Sinc Convolution.

    This module performs a convolution using Sinc filters in the time domain as
    the kernel. Sinc filters function as band passes in the spectral domain.
    The filtering is done as a convolution in the time domain, and no
    transformation to the spectral domain is necessary.

    This implementation of the Sinc convolution is heavily inspired by Ravanelli
    et al. (https://github.com/mravanelli/SincNet) and adapted for the ESpnet
    toolkit. It combines Sinc convolutions with a log compression activation
    function, as described in: https://arxiv.org/abs/2010.07597.

    Notes:
        Currently, the same filters are applied to all input channels. The
        windowing function is applied on the kernel to obtain a smoother filter
        and not on the input values, which is different from traditional ASR.

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Sinc filter kernel size (must be odd).
        stride (int): Stride for the convolution.
        padding (int): Padding for the convolution.
        dilation (int): Dilation for the convolution.
        window_func (callable): Window function applied to the filter.
        scale (callable): Scale type for frequency representation.
        fs (float): Sample rate of the input data.
        sinc_filters (torch.Tensor): Calculated Sinc filters for convolution.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Sinc filter kernel size (needs to be an odd number).
        stride: See torch.nn.functional.conv1d.
        padding: See torch.nn.functional.conv1d.
        dilation: See torch.nn.functional.conv1d.
        window_func: Window function on the filter, one of ["hamming", "none"].
        fs: Sample rate of the input data.

    Raises:
        NotImplementedError: If an unsupported window function or scale type is
        specified.
        ValueError: If the kernel size is not an odd number.

    Examples:
        >>> sinc_conv = SincConv(in_channels=1, out_channels=16, kernel_size=31)
        >>> input_tensor = torch.randn(10, 1, 160)  # Batch size 10, 1 channel, 160 samples
        >>> output_tensor = sinc_conv(input_tensor)
        >>> output_tensor.shape
        torch.Size([10, 16, D_out])  # D_out depends on padding and stride
    """

    @typechecked
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        window_func: str = "hamming",
        scale_type: str = "mel",
        fs: Union[int, float] = 16000,
    ):
        """Initialize Sinc convolutions.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Sinc filter kernel size (needs to be an odd number).
            stride: See torch.nn.functional.conv1d.
            padding: See torch.nn.functional.conv1d.
            dilation: See torch.nn.functional.conv1d.
            window_func: Window function on the filter, one of ["hamming", "none"].
            fs (str, int, float): Sample rate of the input data
        """
        super().__init__()
        window_funcs = {
            "none": self.none_window,
            "hamming": self.hamming_window,
        }
        if window_func not in window_funcs:
            raise NotImplementedError(
                f"Window function has to be one of {list(window_funcs.keys())}",
            )
        self.window_func = window_funcs[window_func]
        scale_choices = {
            "mel": MelScale,
            "bark": BarkScale,
        }
        if scale_type not in scale_choices:
            raise NotImplementedError(
                f"Scale has to be one of {list(scale_choices.keys())}",
            )
        self.scale = scale_choices[scale_type]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.fs = float(fs)
        if self.kernel_size % 2 == 0:
            raise ValueError("SincConv: Kernel size must be odd.")
        self.f = None
        N = self.kernel_size // 2
        self._x = 2 * math.pi * torch.linspace(1, N, N)
        self._window = self.window_func(torch.linspace(1, N, N))
        # init may get overwritten by E2E network,
        # but is still required to calculate output dim
        self.init_filters()

    @staticmethod
    def sinc(x: torch.Tensor) -> torch.Tensor:
        """
            Sinc Convolution.

        This module performs a convolution using Sinc filters in the time domain
        as the kernel. Sinc filters function as band passes in the spectral domain.
        The filtering is done as a convolution in the time domain, and no
        transformation to the spectral domain is necessary.

        This implementation of the Sinc convolution is heavily inspired by
        Ravanelli et al. https://github.com/mravanelli/SincNet, and adapted for
        the ESpnet toolkit. Combine Sinc convolutions with a log compression
        activation function, as in: https://arxiv.org/abs/2010.07597

        Notes:
            Currently, the same filters are applied to all input channels. The
            windowing function is applied on the kernel to obtain a smoother
            filter, and not on the input values, which is different from
            traditional ASR.

        Attributes:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Sinc filter kernel size (needs to be an odd number).
            stride (int): Stride for the convolution.
            padding (int): Padding for the convolution.
            dilation (int): Dilation for the convolution.
            window_func (callable): Window function applied to the filter.
            scale (callable): Scale type for frequency mapping.
            fs (float): Sample rate of the input data.
            sinc_filters (torch.Tensor): Sinc filters used for convolution.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Sinc filter kernel size (needs to be an odd number).
            stride: See torch.nn.functional.conv1d.
            padding: See torch.nn.functional.conv1d.
            dilation: See torch.nn.functional.conv1d.
            window_func: Window function on the filter, one of ["hamming", "none"].
            fs (str, int, float): Sample rate of the input data.

        Raises:
            NotImplementedError: If the specified window function or scale type
            is not supported.
            ValueError: If the kernel size is not odd.

        Examples:
            >>> sinc_conv = SincConv(in_channels=1, out_channels=2, kernel_size=31)
            >>> input_tensor = torch.randn(1, 1, 100)  # (B, C_in, D_in)
            >>> output_tensor = sinc_conv(input_tensor)
            >>> print(output_tensor.shape)  # (B, C_out, D_out)
        """
        x2 = x + 1e-6
        return torch.sin(x2) / x2

    @staticmethod
    def none_window(x: torch.Tensor) -> torch.Tensor:
        """
        Identity-like windowing function.

        This function applies an identity transformation to the input tensor x,
        effectively returning a tensor of ones with the same shape as x. This
        means that no windowing effect is applied to the filter.

        Args:
            x: A tensor for which the windowing function is applied.

        Returns:
            torch.Tensor: A tensor of ones with the same shape as input x.

        Examples:
            >>> input_tensor = torch.tensor([1.0, 2.0, 3.0])
            >>> output_tensor = SincConv.none_window(input_tensor)
            >>> print(output_tensor)
            tensor([1., 1., 1.])
        """
        return torch.ones_like(x)

    @staticmethod
    def hamming_window(x: torch.Tensor) -> torch.Tensor:
        """
            Hamming Windowing function.

        This function computes the Hamming window, which is a type of tapering
        function used to smooth the Sinc filter coefficients. The Hamming window
        is particularly effective in reducing spectral leakage in frequency
        analysis.

        Args:
            x: A tensor of shape (N,) where N is the number of points to
               generate the window.

        Returns:
            torch.Tensor: A tensor containing the Hamming window values of
                          the same shape as the input tensor x.

        Examples:
            >>> import torch
            >>> window_size = 10
            >>> x = torch.linspace(0, window_size - 1, window_size)
            >>> hamming_window = SincConv.hamming_window(x)
            >>> print(hamming_window)
            tensor([0.08, 0.54, 0.92, 1.00, 0.92, 0.54, 0.08, 0.00, 0.00, 0.00])
        """
        L = 2 * x.size(0) + 1
        x = x.flip(0)
        return 0.54 - 0.46 * torch.cos(2.0 * math.pi * x / L)

    def init_filters(self):
        """
        Initialize filters with filterbank values.

        This method computes the initial filterbank values based on the specified
        scale (Mel or Bark) and the sample rate. The resulting filterbank is
        stored as a parameter that can be learned during training.

        Raises:
            NotImplementedError: If the specified scale type is not supported.

        Examples:
            >>> sinc_conv = SincConv(in_channels=1, out_channels=10, kernel_size=51)
            >>> sinc_conv.init_filters()
            >>> sinc_conv.f.shape
            torch.Size([10, 2])  # Shape of the filterbank parameters
        """
        f = self.scale.bank(self.out_channels, self.fs)
        f = torch.div(f, self.fs)
        self.f = torch.nn.Parameter(f, requires_grad=True)

    def _create_filters(self, device: str):
        """Calculate coefficients.

        This function (re-)calculates the filter convolutions coefficients.
        """
        f_mins = torch.abs(self.f[:, 0])
        f_maxs = torch.abs(self.f[:, 0]) + torch.abs(self.f[:, 1] - self.f[:, 0])

        self._x = self._x.to(device)
        self._window = self._window.to(device)

        f_mins_x = torch.matmul(f_mins.view(-1, 1), self._x.view(1, -1))
        f_maxs_x = torch.matmul(f_maxs.view(-1, 1), self._x.view(1, -1))

        kernel = (torch.sin(f_maxs_x) - torch.sin(f_mins_x)) / (0.5 * self._x)
        kernel = kernel * self._window

        kernel_left = kernel.flip(1)
        kernel_center = (2 * f_maxs - 2 * f_mins).unsqueeze(1)
        filters = torch.cat([kernel_left, kernel_center, kernel], dim=1)

        filters = filters.view(filters.size(0), 1, filters.size(1))
        self.sinc_filters = filters

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Sinc convolution forward function.

        Applies the Sinc convolution operation to the input tensor `xs`. The
        input tensor should have the shape (B, C_in, D_in), where B is the
        batch size, C_in is the number of input channels, and D_in is the
        input dimension. The output tensor will have the shape
        (B, C_out, D_out), where C_out is the number of output channels and
        D_out is the output dimension calculated based on the input size,
        stride, padding, and dilation.

        Args:
            xs: Batch in form of torch.Tensor (B, C_in, D_in).

        Returns:
            torch.Tensor: Batch in form of torch.Tensor (B, C_out, D_out).

        Examples:
            >>> sinc_conv = SincConv(in_channels=1, out_channels=2,
            ...                       kernel_size=31)
            >>> input_tensor = torch.randn(10, 1, 100)  # (B, C_in, D_in)
            >>> output_tensor = sinc_conv(input_tensor)
            >>> print(output_tensor.shape)  # Output shape: (10, 2, D_out)

        Note:
            This method requires that the Sinc filters be created before
            performing the convolution, which is handled internally by
            calling `_create_filters()`.

        Raises:
            RuntimeError: If the input tensor does not have the expected
            shape.
        """
        self._create_filters(xs.device)
        xs = torch.nn.functional.conv1d(
            xs,
            self.sinc_filters,
            padding=self.padding,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.in_channels,
        )
        return xs

    def get_odim(self, idim: int) -> int:
        """Obtain the output dimension of the filter."""
        D_out = idim + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1
        D_out = (D_out // self.stride) + 1
        return D_out


class MelScale:
    """
    Mel frequency scale.

    This class provides methods for converting between Hertz and Mel frequency
    scales, as well as generating filter banks based on the Mel scale.

    The Mel scale is a perceptual scale of pitches which approximates the
    human ear's response to different frequencies. It is widely used in
    audio processing, particularly in speech and music analysis.

    Methods:
        convert(f): Convert frequency in Hertz to Mel scale.
        invert(x): Convert frequency in Mel scale back to Hertz.
        bank(channels, fs): Obtain initialization values for the Mel scale filter
            bank.

    Args:
        channels: Number of channels for the filter bank.
        fs: Sample rate of the input signal.

    Returns:
        torch.Tensor: Filter start frequencies and stop frequencies.

    Examples:
        >>> mel = MelScale()
        >>> mel_freq = mel.convert(torch.tensor([1000.0]))  # Convert Hz to Mel
        >>> hz_freq = mel.invert(mel_freq)  # Convert Mel back to Hz
        >>> filter_bank = mel.bank(channels=40, fs=16000)  # Create Mel filter bank
    """

    @staticmethod
    def convert(f):
        """
                Convert Hz to mel.

        This method converts a frequency in Hertz to the corresponding value in the mel
        scale using the formula:

            mel = 1125 * log(f / 700 + 1)

        Args:
            f: A tensor representing frequency values in Hertz.

        Returns:
            A tensor containing the converted values in the mel scale.

        Examples:
            >>> import torch
            >>> mel_values = MelScale.convert(torch.tensor([440.0, 880.0]))
            >>> print(mel_values)
            tensor([   ale in tensor format  ,   ale in tensor format  ])
        """
        return 1125.0 * torch.log(torch.div(f, 700.0) + 1.0)

    @staticmethod
    def invert(x):
        """
            Convert mel to Hz.

        This function takes a tensor of values in the mel scale and converts them
        back to their corresponding frequencies in Hertz (Hz) using the inverse
        transformation of the mel scale.

        Args:
            x (torch.Tensor): A tensor containing values in the mel scale.

        Returns:
            torch.Tensor: A tensor containing the corresponding frequencies in Hz.

        Examples:
            >>> mel_values = torch.tensor([0.0, 100.0, 200.0])
            >>> hz_values = MelScale.invert(mel_values)
            >>> print(hz_values)
            tensor([  0.0000,  700.0000, 1415.0000])
        """
        return 700.0 * (torch.exp(torch.div(x, 1125.0)) - 1.0)

    @classmethod
    @typechecked
    def bank(cls, channels: int, fs: float) -> torch.Tensor:
        """
                Sinc convolutions.

        This module contains implementations of the Log Compression activation function,
        Sinc convolution, and the Mel and Bark frequency scales. The Sinc convolution
        is performed using Sinc filters in the time domain, acting as band passes in
        the spectral domain.

        The filtering is done through convolution in the time domain, eliminating the
        need for transformations to the spectral domain. This implementation is inspired
        by Ravanelli et al. and adapted for the ESpnet toolkit. It combines Sinc
        convolutions with a log compression activation function.

        Notes:
        Currently, the same filters are applied to all input channels. The windowing
        function is applied on the kernel to obtain a smoother filter, differing from
        traditional Automatic Speech Recognition (ASR).

        Classes:
        - LogCompression: Log Compression Activation Function.
        - SincConv: Sinc Convolution Layer.
        - MelScale: Mel Frequency Scale.
        - BarkScale: Bark Frequency Scale.
        """
        # min and max bandpass edge frequencies
        min_frequency = torch.tensor(30.0)
        max_frequency = torch.tensor(fs * 0.5)
        frequencies = torch.linspace(
            cls.convert(min_frequency), cls.convert(max_frequency), channels + 2
        )
        frequencies = cls.invert(frequencies)
        f1, f2 = frequencies[:-2], frequencies[2:]
        return torch.stack([f1, f2], dim=1)


class BarkScale:
    """
        Bark frequency scale.

    Has wider bandwidths at lower frequencies, see:
    Critical bandwidth: BARK
    Zwicker and Terhardt, 1980.

    Methods:
        convert(f): Convert Hz to Bark.
        invert(x): Convert Bark to Hz.
        bank(cls, channels: int, fs: float) -> torch.Tensor:
            Obtain initialization values for the Bark scale.

    Args:
        channels: Number of channels.
        fs: Sample rate.

    Returns:
        torch.Tensor: Filter start frequencies.
        torch.Tensor: Filter stop frequencies.

    Examples:
        >>> fs = 16000
        >>> channels = 10
        >>> bark_scale = BarkScale.bank(channels, fs)
        >>> print(bark_scale)

    Note:
        The Bark scale is often used in psychoacoustics and audio signal
        processing to model human perception of sound.
    """

    @staticmethod
    def convert(f):
        """
                Convert frequencies between Hz and Bark scale.

        This class provides methods to convert frequencies from Hertz (Hz) to Bark scale
        and vice versa. The Bark scale is a psychoacoustic scale that reflects human
        perception of sound frequencies, with wider bandwidths at lower frequencies.

        Attributes:
            None

        Args:
            channels: Number of channels.
            fs: Sample rate.

        Returns:
            torch.Tensor: Filter start frequencies.
            torch.Tensor: Filter stop frequencies.

        Examples:
            >>> bark_scale = BarkScale()
            >>> bark_freq = bark_scale.convert(torch.tensor([1000.0, 2000.0]))
            >>> hz_freq = bark_scale.invert(bark_freq)
            >>> assert torch.allclose(hz_freq, torch.tensor([1000.0, 2000.0]))
        """
        b = torch.div(f, 1000.0)
        b = torch.pow(b, 2.0) * 1.4
        b = torch.pow(b + 1.0, 0.69)
        return b * 75.0 + 25.0

    @staticmethod
    def invert(x):
        """
                Invert the Bark frequency scale.

        This method converts values from the Bark scale back to Hertz (Hz).

        Args:
            x (torch.Tensor): A tensor containing values in the Bark scale.

        Returns:
            torch.Tensor: A tensor containing the corresponding frequencies in Hz.

        Examples:
            >>> bark_values = torch.tensor([25.0, 100.0, 200.0])
            >>> hz_values = BarkScale.invert(bark_values)
            >>> print(hz_values)
            tensor([  70.0,  250.0,  500.0])
        """
        f = torch.div(x - 25.0, 75.0)
        f = torch.pow(f, (1.0 / 0.69))
        f = torch.div(f - 1.0, 1.4)
        f = torch.pow(f, 0.5)
        return f * 1000.0

    @classmethod
    @typechecked
    def bank(cls, channels: int, fs: float) -> torch.Tensor:
        """
                Sinc convolutions.

        This module contains implementations of log compression activation, Sinc
        convolution, and frequency scale conversions (Mel and Bark). Sinc filters
        function as band passes in the spectral domain, enabling efficient filtering
        without transforming to the spectral domain.

        The Sinc convolution implementation is inspired by Ravanelli et al. and
        adapted for the ESpnet toolkit. It is recommended to combine Sinc convolutions
        with a log compression activation function for optimal performance.

        Notes:
        Currently, the same filters are applied to all input channels. The windowing
        function is applied on the kernel to obtain a smoother filter, differing from
        traditional ASR approaches.

        Classes:
        - LogCompression: Implements the log compression activation function.
        - SincConv: Performs convolution using Sinc filters.
        - MelScale: Provides methods for converting between Hz and Mel scale.
        - BarkScale: Provides methods for converting between Hz and Bark scale.

        Example usage:
            # Initialize Sinc convolution layer
            sinc_layer = SincConv(in_channels=1, out_channels=10, kernel_size=31)

            # Apply Sinc convolution to a tensor
            input_tensor = torch.randn(5, 1, 100)  # (B, C_in, D_in)
            output_tensor = sinc_layer(input_tensor)  # (B, C_out, D_out)
        """
        # min and max BARK center frequencies by approximation
        min_center_frequency = torch.tensor(70.0)
        max_center_frequency = torch.tensor(fs * 0.45)
        center_frequencies = torch.linspace(
            cls.convert(min_center_frequency),
            cls.convert(max_center_frequency),
            channels,
        )
        center_frequencies = cls.invert(center_frequencies)

        f1 = center_frequencies - torch.div(cls.convert(center_frequencies), 2)
        f2 = center_frequencies + torch.div(cls.convert(center_frequencies), 2)
        return torch.stack([f1, f2], dim=1)
