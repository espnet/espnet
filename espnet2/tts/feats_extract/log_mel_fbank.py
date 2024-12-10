from typing import Any, Dict, Optional, Tuple, Union

import humanfriendly
import torch
from typeguard import typechecked

from espnet2.layers.log_mel import LogMel
from espnet2.layers.stft import Stft
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract


class LogMelFbank(AbsFeatsExtract):
    """
    LogMelFbank is a conventional frontend structure for Text-to-Speech (TTS)
    systems. It processes audio signals through Short-Time Fourier Transform (STFT)
    to produce log-mel filter bank features.

    The processing flow is as follows:
    STFT -> amplitude-spectrum -> Log-Mel-Fbank

    Attributes:
        fs (Union[int, str]): Sampling frequency. Can be an integer or a string
            representing size (e.g., "16k").
        n_fft (int): Number of FFT points.
        win_length (Optional[int]): Window length for STFT. If None, defaults to
            n_fft.
        hop_length (int): Hop length for STFT.
        window (Optional[str]): Window function type (e.g., "hann").
        center (bool): Whether to pad the signal on both sides so that the
            frame is centered at the point.
        normalized (bool): Whether to normalize the STFT output.
        onesided (bool): Whether to return a one-sided spectrum.
        n_mels (int): Number of Mel bands to generate.
        fmin (Optional[int]): Minimum frequency (in Hz) to consider.
        fmax (Optional[int]): Maximum frequency (in Hz) to consider.
        htk (bool): Whether to use HTK formula for Mel scale.
        log_base (Optional[float]): Base of the logarithm for log-mel scaling.

    Args:
        fs (Union[int, str]): Sampling frequency. Default is 16000.
        n_fft (int): Number of FFT points. Default is 1024.
        win_length (Optional[int]): Window length. Default is None.
        hop_length (int): Hop length. Default is 256.
        window (Optional[str]): Type of window function. Default is "hann".
        center (bool): Centering of the window. Default is True.
        normalized (bool): Normalization of the output. Default is False.
        onesided (bool): One-sided spectrum output. Default is True.
        n_mels (int): Number of Mel bands. Default is 80.
        fmin (Optional[int]): Minimum frequency. Default is 80.
        fmax (Optional[int]): Maximum frequency. Default is 7600.
        htk (bool): HTK formula usage. Default is False.
        log_base (Optional[float]): Logarithm base. Default is 10.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - output features (torch.Tensor): The log-mel features.
            - feats_lens (torch.Tensor): The lengths of the features.

    Examples:
        >>> logmel_fbank = LogMelFbank()
        >>> audio_input = torch.randn(1, 16000)  # Simulated audio input
        >>> features, lengths = logmel_fbank.forward(audio_input)

    Note:
        The TTS definition for log-spectral features differs from ASR. TTS uses
        log_10(abs(stft)), while ASR uses log_e(power(stft)).

    Raises:
        AssertionError: If the input STFT tensor does not have the expected
        dimensions or shape.
    """

    @typechecked
    def __init__(
        self,
        fs: Union[int, str] = 16000,
        n_fft: int = 1024,
        win_length: Optional[int] = None,
        hop_length: int = 256,
        window: Optional[str] = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        n_mels: int = 80,
        fmin: Optional[int] = 80,
        fmax: Optional[int] = 7600,
        htk: bool = False,
        log_base: Optional[float] = 10.0,
    ):
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)

        self.fs = fs
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.fmin = fmin
        self.fmax = fmax

        self.stft = Stft(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=window,
            center=center,
            normalized=normalized,
            onesided=onesided,
        )

        self.logmel = LogMel(
            fs=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
            log_base=log_base,
        )

    def output_size(self) -> int:
        """
            Returns the number of Mel frequency bands.

        This property retrieves the number of Mel frequency bands configured for
        the LogMelFbank instance. It is primarily used to determine the size of
        the output feature representations.

        Returns:
            int: The number of Mel frequency bands.

        Examples:
            logmel_fbank = LogMelFbank(n_mels=80)
            output_size = logmel_fbank.output_size()
            print(output_size)  # Output: 80
        """
        return self.n_mels

    def get_parameters(self) -> Dict[str, Any]:
        """
        Return the parameters required by Vocoder.

        This method gathers and returns a dictionary of parameters that are
        essential for the vocoder to operate effectively. These parameters
        include sampling frequency, FFT size, hop length, window type, number
        of Mel filters, window length, minimum frequency, and maximum frequency.

        Returns:
            Dict[str, Any]: A dictionary containing the following keys and their
            corresponding values:
                - fs (int): Sampling frequency.
                - n_fft (int): Number of FFT points.
                - n_shift (int): Hop length (number of samples between frames).
                - window (str): Type of windowing function used.
                - n_mels (int): Number of Mel bands.
                - win_length (Optional[int]): Length of the window.
                - fmin (Optional[int]): Minimum frequency (in Hz).
                - fmax (Optional[int]): Maximum frequency (in Hz).

        Examples:
            >>> logmel_fbank = LogMelFbank()
            >>> params = logmel_fbank.get_parameters()
            >>> print(params)
            {'fs': 16000, 'n_fft': 1024, 'n_shift': 256,
             'window': 'hann', 'n_mels': 80,
             'win_length': None, 'fmin': 80, 'fmax': 7600}
        """
        return dict(
            fs=self.fs,
            n_fft=self.n_fft,
            n_shift=self.hop_length,
            window=self.window,
            n_mels=self.n_mels,
            win_length=self.win_length,
            fmin=self.fmin,
            fmax=self.fmax,
        )

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Computes the Log-Mel filterbank features from the input audio tensor.

        This method performs a sequence of operations on the input audio tensor,
        including Short-Time Fourier Transform (STFT), converting the complex
        spectrogram to amplitude, and then applying the Log-Mel filterbank to
        extract features suitable for speech synthesis.

        Args:
            input (torch.Tensor): The input audio tensor of shape (..., T),
                where T is the number of time frames.
            input_lengths (torch.Tensor, optional): A tensor containing the lengths
                of the input sequences. If not provided, it defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - A tensor of shape (..., n_mels, T') representing the extracted
                  Log-Mel features, where T' is the number of output time frames.
                - A tensor containing the lengths of the output features.

        Raises:
            AssertionError: If the input STFT tensor does not have at least 4
                dimensions or if the last dimension is not equal to 2.

        Note:
            The Log-Mel computation uses a different definition of log-spectra
            between Text-to-Speech (TTS) and Automatic Speech Recognition (ASR):
            - TTS: log_10(abs(stft))
            - ASR: log_e(power(stft))

        Examples:
            >>> logmel_fbank = LogMelFbank()
            >>> audio_input = torch.randn(1, 16000)  # Example audio input
            >>> features, lengths = logmel_fbank.forward(audio_input)
            >>> print(features.shape)  # Output shape will be (..., n_mels, T')
        """
        # 1. Domain-conversion: e.g. Stft: time -> time-freq
        input_stft, feats_lens = self.stft(input, input_lengths)

        assert input_stft.dim() >= 4, input_stft.shape
        # "2" refers to the real/imag parts of Complex
        assert input_stft.shape[-1] == 2, input_stft.shape

        # NOTE(kamo): We use different definition for log-spec between TTS and ASR
        #   TTS: log_10(abs(stft))
        #   ASR: log_e(power(stft))

        # input_stft: (..., F, 2) -> (..., F)
        input_power = input_stft[..., 0] ** 2 + input_stft[..., 1] ** 2
        input_amp = torch.sqrt(torch.clamp(input_power, min=1.0e-10))
        input_feats, _ = self.logmel(input_amp, feats_lens)
        return input_feats, feats_lens
