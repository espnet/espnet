from typing import Any, Dict, Optional, Tuple

import torch
from typeguard import typechecked

from espnet2.layers.stft import Stft
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract


class LogSpectrogram(AbsFeatsExtract):
    """
    LogSpectrogram is a conventional frontend structure for Automatic Speech
    Recognition (ASR) that converts time-domain audio signals into log-amplitude
    spectrograms using Short-Time Fourier Transform (STFT).

    The transformation pipeline consists of:
    1. STFT: converting time-domain signals to time-frequency representation.
    2. Log-amplitude spectrum: calculating the logarithm of the amplitude of the
       resulting frequency bins.

    Attributes:
        n_fft (int): Number of FFT points.
        hop_length (int): Number of samples between frames.
        win_length (Optional[int]): Length of the windowed signal.
        window (Optional[str]): Type of window to apply.
        stft (Stft): Instance of the STFT class for performing STFT.

    Args:
        n_fft (int): Number of FFT points (default is 1024).
        win_length (Optional[int]): Length of the windowed signal (default is None).
        hop_length (int): Number of samples between frames (default is 256).
        window (Optional[str]): Type of window to apply (default is "hann").
        center (bool): If True, the signal is padded so that the frame is centered
                       at the original time index (default is True).
        normalized (bool): If True, the output is normalized (default is False).
        onesided (bool): If True, only the positive half of the spectrum is returned
                         (default is True).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the log-amplitude
        spectrogram and the lengths of the features.

    Examples:
        >>> log_spectrogram = LogSpectrogram(n_fft=2048, hop_length=512)
        >>> audio_input = torch.randn(1, 16000)  # Simulated audio input
        >>> log_amp, feats_lens = log_spectrogram(audio_input)

    Note:
        The log-spectrogram is defined differently between TTS and ASR:
        - TTS: log_10(abs(stft))
        - ASR: log_e(power(stft))

    Raises:
        AssertionError: If the input STFT tensor does not have the expected
        dimensions or shape.

    Todo:
        Consider adding more configurable parameters for window functions and
        normalization methods in future versions.
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
            Calculate the output size of the LogSpectrogram.

        The output size is computed based on the number of FFT points (n_fft) used
        in the Short-Time Fourier Transform (STFT). The output size represents the
        number of frequency bins in the log-amplitude spectrogram, which is given
        by the formula `n_fft // 2 + 1`.

        Returns:
            int: The number of frequency bins in the log-amplitude spectrogram.

        Examples:
            >>> log_spectrogram = LogSpectrogram(n_fft=1024)
            >>> log_spectrogram.output_size()
            513
        """
        return self.n_fft // 2 + 1

    def get_parameters(self) -> Dict[str, Any]:
        """
        Returns the parameters required by the Vocoder.

        This method gathers the essential parameters used for the vocoder,
        which include the number of FFT points, the hop length, the window
        length, and the window type. These parameters are crucial for
        generating the correct spectrogram representation needed by the vocoder.

        Returns:
            A dictionary containing the following key-value pairs:
                - n_fft (int): Number of FFT points.
                - n_shift (int): Hop length (number of samples to shift).
                - win_length (Optional[int]): Window length (if specified).
                - window (Optional[str]): Type of window used for STFT.

        Examples:
            >>> log_spectrogram = LogSpectrogram(n_fft=2048, hop_length=512)
            >>> parameters = log_spectrogram.get_parameters()
            >>> print(parameters)
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
            Computes the log-amplitude spectrogram from the input audio tensor.

        This method takes an input tensor representing audio signals and converts
        it into a log-amplitude spectrogram using Short-Time Fourier Transform (STFT).
        The output consists of the log-amplitude features and their corresponding
        lengths.

        Args:
            input (torch.Tensor): The input audio tensor with shape
                (batch_size, num_samples).
            input_lengths (torch.Tensor, optional): A tensor containing the lengths
                of the input sequences. If None, all sequences are assumed to have
                the same length.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - log_amp (torch.Tensor): The computed log-amplitude spectrogram
                  with shape (batch_size, num_features, time_steps).
                - feats_lens (torch.Tensor): The lengths of the output features
                  after processing.

        Raises:
            AssertionError: If the input STFT tensor does not have the expected
            dimensions or if the last dimension does not represent real/imaginary
            parts.

        Examples:
            >>> model = LogSpectrogram(n_fft=1024)
            >>> input_tensor = torch.randn(4, 16000)  # Example batch of audio
            >>> log_amp, lengths = model.forward(input_tensor)

        Note:
            The log-amplitude is computed using log_10(abs(stft)) for TTS
            applications and log_e(power(stft)) for ASR applications.
        """
        # 1. Stft: time -> time-freq
        input_stft, feats_lens = self.stft(input, input_lengths)

        assert input_stft.dim() >= 4, input_stft.shape
        # "2" refers to the real/imag parts of Complex
        assert input_stft.shape[-1] == 2, input_stft.shape

        # NOTE(kamo): We use different definition for log-spec between TTS and ASR
        #   TTS: log_10(abs(stft))
        #   ASR: log_e(power(stft))

        # STFT -> Power spectrum
        # input_stft: (..., F, 2) -> (..., F)
        input_power = input_stft[..., 0] ** 2 + input_stft[..., 1] ** 2
        log_amp = 0.5 * torch.log10(torch.clamp(input_power, min=1.0e-10))
        return log_amp, feats_lens
