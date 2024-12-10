from typing import Any, Dict, Optional, Tuple

import torch
from typeguard import typechecked

from espnet2.layers.stft import Stft
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract


class LinearSpectrogram(AbsFeatsExtract):
    """
        Linear amplitude spectrogram.

    This class computes the linear amplitude spectrogram from an input signal
    using Short-Time Fourier Transform (STFT). It inherits from the
    `AbsFeatsExtract` abstract class.

    Attributes:
        n_fft (int): Number of FFT points.
        hop_length (int): Number of samples between adjacent STFT columns.
        win_length (Optional[int]): Window length. If None, it defaults to `n_fft`.
        window (Optional[str]): Window function type. Defaults to "hann".
        stft (Stft): STFT instance used for transforming the input signal.

    Args:
        n_fft (int, optional): Number of FFT points. Defaults to 1024.
        win_length (Optional[int], optional): Window length. Defaults to None.
        hop_length (int, optional): Number of samples between STFT columns.
            Defaults to 256.
        window (Optional[str], optional): Window type. Defaults to "hann".
        center (bool, optional): If True, the signal is padded so that
            the t-th frame is centered at time t. Defaults to True.
        normalized (bool, optional): If True, the output will be normalized.
            Defaults to False.
        onesided (bool, optional): If True, the output will be one-sided.
            Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the amplitude
        spectrogram and the lengths of the features.

    Examples:
        # Create a LinearSpectrogram instance
        spectrogram = LinearSpectrogram(n_fft=2048, hop_length=512)

        # Forward pass with an input tensor
        input_tensor = torch.randn(1, 16000)  # Example input
        amp_spectrogram, lengths = spectrogram(input_tensor)

    Note:
        The input tensor should have dimensions (batch_size, num_samples).

    Todo:
        - Add support for additional window types.
    """

    @typechecked
    def __init__(
        self,
        n_fft: int = 1024,
        win_length: Optional[int] = None,
        hop_length: int = 256,
        window: Optional[str] = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.stft = Stft(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=window,
            center=center,
            normalized=normalized,
            onesided=onesided,
        )
        self.n_fft = n_fft

    def output_size(self) -> int:
        """
            Returns the output size of the linear spectrogram, which is calculated as
        half of the FFT size plus one. This value represents the number of frequency
        bins in the output spectrogram.

        Returns:
            int: The number of frequency bins in the output spectrogram.

        Examples:
            >>> spectrogram = LinearSpectrogram(n_fft=1024)
            >>> spectrogram.output_size()
            513
        """
        return self.n_fft // 2 + 1

    def get_parameters(self) -> Dict[str, Any]:
        """
        Return the parameters required by Vocoder.

        This method returns a dictionary containing the key parameters
        necessary for the vocoder, which include the number of FFT points,
        the hop length, the window length, and the window type.

        Returns:
            A dictionary with the following keys:
                - n_fft (int): The number of FFT points.
                - n_shift (int): The hop length.
                - win_length (Optional[int]): The window length.
                - window (Optional[str]): The window type.

        Examples:
            >>> spectrogram = LinearSpectrogram(n_fft=2048, hop_length=512)
            >>> params = spectrogram.get_parameters()
            >>> print(params)
            {'n_fft': 2048, 'n_shift': 512, 'win_length': None, 'window': 'hann'}
        """
        return dict(
            n_fft=self.n_fft,
            n_shift=self.hop_length,
            win_length=self.win_length,
            window=self.window,
        )

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Computes the linear amplitude spectrogram from the input tensor using Short-Time
        Fourier Transform (STFT).

        The `forward` method takes an input tensor, applies STFT to convert the time
        domain signal into the frequency domain, and then computes the amplitude
        spectrum from the resulting complex spectrogram.

        Args:
            input (torch.Tensor): Input tensor of shape (..., T), where T is the
                number of time steps. This tensor represents the audio waveform.
            input_lengths (torch.Tensor, optional): A tensor containing the lengths of
                the input sequences. This is useful for batching inputs of varying lengths.
                If not provided, the method assumes all inputs are of the same length.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - A tensor of shape (..., F) representing the amplitude spectrum,
                  where F is the number of frequency bins.
                - A tensor of shape (...,) representing the lengths of the features
                  after applying STFT.

        Raises:
            AssertionError: If the input tensor's dimension is less than 4 or if
                the last dimension of the input tensor is not equal to 2.

        Examples:
            >>> linear_spectrogram = LinearSpectrogram(n_fft=1024)
            >>> input_tensor = torch.randn(1, 16000)  # Example input
            >>> amp_spectrum, feats_lengths = linear_spectrogram.forward(input_tensor)
            >>> print(amp_spectrum.shape)  # Output shape will be (..., F)
            >>> print(feats_lengths.shape)  # Output shape will be (...,)
        """
        # 1. Stft: time -> time-freq
        input_stft, feats_lens = self.stft(input, input_lengths)

        assert input_stft.dim() >= 4, input_stft.shape
        # "2" refers to the real/imag parts of Complex
        assert input_stft.shape[-1] == 2, input_stft.shape

        # STFT -> Power spectrum -> Amp spectrum
        # input_stft: (..., F, 2) -> (..., F)
        input_power = input_stft[..., 0] ** 2 + input_stft[..., 1] ** 2
        input_amp = torch.sqrt(torch.clamp(input_power, min=1.0e-10))
        return input_amp, feats_lens
