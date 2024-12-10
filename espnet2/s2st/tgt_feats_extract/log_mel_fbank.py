from typing import Any, Dict, Optional, Tuple, Union

import humanfriendly
import torch
from typeguard import typechecked

from espnet2.layers.log_mel import LogMel
from espnet2.layers.stft import Stft
from espnet2.s2st.tgt_feats_extract.abs_tgt_feats_extract import AbsTgtFeatsExtract


class LogMelFbank(AbsTgtFeatsExtract):
    """
    LogMelFbank is a conventional frontend structure for Text-to-Speech (TTS)
    systems. It processes audio input through a series of transformations
    including Short-Time Fourier Transform (STFT), amplitude spectrum
    computation, and finally, conversion to Log-Mel filterbank features.

    The sequence of operations is as follows:
    - STFT: Converts time-domain signal to time-frequency domain.
    - Amplitude-Spec: Computes the amplitude from the STFT.
    - Log-Mel-Fbank: Applies a Log-Mel filterbank to the amplitude spectrum.

    Attributes:
        fs (int): Sampling frequency of the input audio.
        n_mels (int): Number of Mel bands to generate.
        n_fft (int): Size of the FFT window.
        hop_length (int): Number of samples between frames.
        win_length (Optional[int]): Length of the windowed signal.
        window (Optional[str]): Type of window function to apply.
        fmin (Optional[int]): Minimum frequency (in Hz) to consider.
        fmax (Optional[int]): Maximum frequency (in Hz) to consider.
        stft (Stft): Instance of the STFT class for time-frequency conversion.
        logmel (LogMel): Instance of the LogMel class for Mel feature extraction.

    Args:
        fs (Union[int, str]): Sampling frequency (default is 16000).
        n_fft (int): Size of the FFT window (default is 1024).
        win_length (Optional[int]): Length of the windowed signal (default is None).
        hop_length (int): Number of samples between frames (default is 256).
        window (Optional[str]): Type of window function to apply (default is "hann").
        center (bool): If True, the signal is padded so that the window is
            centered at the current frame (default is True).
        normalized (bool): If True, the output is normalized (default is False).
        onesided (bool): If True, only the positive frequencies are returned
            (default is True).
        n_mels (int): Number of Mel bands to generate (default is 80).
        fmin (Optional[int]): Minimum frequency (in Hz) to consider (default is 80).
        fmax (Optional[int]): Maximum frequency (in Hz) to consider (default is 7600).
        htk (bool): If True, use HTK formula for Mel scale (default is False).
        log_base (Optional[float]): Base of the logarithm (default is 10.0).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - torch.Tensor: Extracted Log-Mel features.
            - torch.Tensor: Lengths of the features for each input.

    Examples:
        # Create an instance of LogMelFbank
        log_mel_fbank = LogMelFbank(fs=16000, n_mels=80)

        # Forward pass with a sample input tensor
        input_tensor = torch.randn(1, 16000)  # Example input
        features, lengths = log_mel_fbank(input_tensor)

    Note:
        The implementation assumes that the input tensor is in the shape
        (batch_size, time) for single-channel audio.

    Raises:
        AssertionError: If the input STFT does not have the expected dimensions.
    """

    @typechecked
    def __init__(
        self,
        fs: Union[int, str] = 16000,
        n_fft: int = 1024,
        win_length: int = None,
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
            Get the output size of the LogMelFbank feature extractor.

        This property returns the number of Mel frequency bins used in the
        Log-Mel-Fbank representation.

        Returns:
            int: The number of Mel frequency bins (n_mels) configured in the
            LogMelFbank instance.

        Examples:
            logmel_fbank = LogMelFbank(n_mels=80)
            output_size = logmel_fbank.output_size()  # output_size will be 80

        Note:
            The output size corresponds to the `n_mels` parameter set during
            the initialization of the LogMelFbank class.
        """
        return self.n_mels

    def get_parameters(self) -> Dict[str, Any]:
        """
        Return the parameters required by Vocoder.

        This method retrieves and returns a dictionary containing
        the configuration parameters for the vocoder. These parameters
        are essential for the vocoder to process audio data correctly.

        Returns:
            A dictionary with the following keys and their corresponding values:
                - fs: Sampling frequency (int)
                - n_fft: Number of FFT points (int)
                - n_shift: Hop length for the STFT (int)
                - window: Window type used for STFT (str)
                - n_mels: Number of Mel bands (int)
                - win_length: Window length for STFT (int or None)
                - fmin: Minimum frequency (int or None)
                - fmax: Maximum frequency (int or None)

        Examples:
            >>> logmel_fbank = LogMelFbank()
            >>> parameters = logmel_fbank.get_parameters()
            >>> print(parameters)
            {'fs': 16000, 'n_fft': 1024, 'n_shift': 256,
             'window': 'hann', 'n_mels': 80, 'win_length': None,
             'fmin': 80, 'fmax': 7600}
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
            Forward pass for the LogMelFbank module.

        This method processes the input audio tensor and converts it into a
        log-mel spectrogram. It first applies Short-Time Fourier Transform (STFT)
        to convert the time-domain signal into the frequency domain, then computes
        the amplitude, and finally applies the log-mel filterbank to produce the
        output features.

        Args:
            input (torch.Tensor): Input audio tensor of shape
                (..., time).
            input_lengths (torch.Tensor, optional): Lengths of the input sequences.
                Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - A tensor of shape (..., n_mels, time) representing the log-mel
                  spectrogram.
                - A tensor containing the lengths of the output features.

        Raises:
            AssertionError: If the input_stft does not have at least 4 dimensions
                or if the last dimension is not equal to 2.

        Examples:
            >>> logmel_fbank = LogMelFbank()
            >>> audio_input = torch.randn(1, 16000)  # Simulated audio input
            >>> features, lengths = logmel_fbank.forward(audio_input)
            >>> print(features.shape)  # Output shape will be (..., n_mels, time)

        Note:
            The log-mel computation is defined differently for TTS and ASR:
            - TTS: log_10(abs(stft))
            - ASR: log_e(power(stft))
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

    def spectrogram(self) -> bool:
        """
            Conventional frontend structure for TTS.

        Stft -> amplitude-spec -> Log-Mel-Fbank

        Attributes:
            fs (Union[int, str]): Sampling frequency of the audio signal.
            n_fft (int): Number of FFT points.
            win_length (int, optional): Length of each windowed segment.
            hop_length (int): Number of samples between adjacent frames.
            window (Optional[str]): Window type to apply (default is "hann").
            center (bool): Whether to pad the input signal on both sides.
            normalized (bool): Whether to normalize the output.
            onesided (bool): Whether to use a one-sided spectrum.
            n_mels (int): Number of Mel bands to generate.
            fmin (Optional[int]): Minimum frequency (default is 80 Hz).
            fmax (Optional[int]): Maximum frequency (default is 7600 Hz).
            htk (bool): Use HTK formula for Mel scale if True.
            log_base (Optional[float]): Base of the logarithm (default is 10.0).

        Args:
            fs (Union[int, str]): Sampling frequency (default is 16000).
            n_fft (int): Number of FFT points (default is 1024).
            win_length (int, optional): Length of each windowed segment.
            hop_length (int): Number of samples between adjacent frames (default is 256).
            window (Optional[str]): Window type (default is "hann").
            center (bool): Whether to pad the input signal on both sides (default is True).
            normalized (bool): Whether to normalize the output (default is False).
            onesided (bool): Whether to use a one-sided spectrum (default is True).
            n_mels (int): Number of Mel bands (default is 80).
            fmin (Optional[int]): Minimum frequency (default is 80).
            fmax (Optional[int]): Maximum frequency (default is 7600).
            htk (bool): Use HTK formula for Mel scale if True (default is False).
            log_base (Optional[float]): Base of the logarithm (default is 10.0).

        Returns:
            None

        Examples:
            logmel = LogMelFbank(fs=16000, n_mels=80)
            print(logmel.output_size())  # Outputs: 80
            params = logmel.get_parameters()
            print(params)  # Outputs: Parameters dictionary for vocoder.

        Note:
            This class is designed to work within the ESPnet framework and
            follows the conventional TTS frontend processing pipeline.

        Raises:
            ValueError: If any of the input parameters are invalid.
        """
        return True
