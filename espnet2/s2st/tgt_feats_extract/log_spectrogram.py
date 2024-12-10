from typing import Any, Dict, Optional, Tuple

import torch
from typeguard import typechecked

from espnet2.layers.stft import Stft
from espnet2.s2st.tgt_feats_extract.abs_tgt_feats_extract import AbsTgtFeatsExtract


class LogSpectrogram(AbsTgtFeatsExtract):
    """
    LogSpectrogram is a conventional frontend structure for Automatic Speech
    Recognition (ASR). It processes audio input to produce a log-amplitude
    spectrogram from the Short-Time Fourier Transform (STFT).

    The main processing steps are as follows:
    1. Apply STFT to convert time-domain signals to the time-frequency domain.
    2. Compute the log-amplitude spectrum from the power spectrum of the STFT.

    Attributes:
        n_fft (int): The number of FFT points.
        hop_length (int): The number of samples between each frame.
        win_length (Optional[int]): The length of the windowed signal.
        window (Optional[str]): The type of window function to use.
        stft (Stft): An instance of the STFT class for performing STFT.

    Args:
        n_fft (int): Number of FFT points (default is 1024).
        win_length (Optional[int]): Length of the windowed signal (default is None).
        hop_length (int): Number of samples between each frame (default is 256).
        window (Optional[str]): Type of window function (default is "hann").
        center (bool): Whether to pad the signal on both sides (default is True).
        normalized (bool): Whether to normalize the output (default is False).
        onesided (bool): Whether to return a one-sided spectrum (default is True).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - log_amplitude (torch.Tensor): The log-amplitude spectrogram.
            - feats_lens (torch.Tensor): The lengths of the features.

    Raises:
        AssertionError: If the input STFT tensor does not have the expected shape.

    Examples:
        >>> log_spec = LogSpectrogram(n_fft=2048, hop_length=512)
        >>> input_tensor = torch.randn(1, 16000)  # Simulated audio input
        >>> log_amp, lengths = log_spec.forward(input_tensor)
        >>> print(log_amp.shape)  # Output shape will depend on input length

    Note:
        The log-amplitude spectrum is defined differently for Text-to-Speech (TTS)
        and ASR. In TTS, it is computed as log_10(abs(stft)), while in ASR, it is
        computed as log_e(power(stft)).
    """

    @typechecked
    def __init__(
        self,
        n_fft: int = 1024,
        win_length: int = None,
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
            Returns the output size of the log spectrogram, which is calculated as half
        the FFT size plus one. This is useful for determining the dimensions of the
        output tensor after applying the Short-Time Fourier Transform (STFT).

        The output size is computed as follows:
            output_size = n_fft // 2 + 1

        Attributes:
            n_fft (int): The number of FFT points used in the STFT.

        Returns:
            int: The output size of the log spectrogram.

        Examples:
            >>> log_spectrogram = LogSpectrogram(n_fft=1024)
            >>> log_spectrogram.output_size()
            513
        """
        return self.n_fft // 2 + 1

    def get_parameters(self) -> Dict[str, Any]:
        """
            Return the parameters required by Vocoder.

        This method gathers the essential parameters used in the
        vocoder process, including the number of FFT points,
        hop length, window length, and window type. These parameters
        are crucial for configuring the vocoder's behavior.

        Returns:
            A dictionary containing the following parameters:
            - n_fft (int): The number of FFT points.
            - n_shift (int): The hop length for the STFT.
            - win_length (Optional[int]): The window length for STFT.
            - window (Optional[str]): The type of window used.

        Examples:
            >>> log_spectrogram = LogSpectrogram(n_fft=2048, hop_length=512)
            >>> params = log_spectrogram.get_parameters()
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
            Forward pass to compute the log-amplitude spectrogram from the input tensor.

        This method performs a Short-Time Fourier Transform (STFT) on the input
        audio tensor, calculates the power spectrum, and then computes the log-amplitude
        spectrogram. The output consists of the log-amplitude features and their
        corresponding lengths.

        Args:
            input (torch.Tensor): The input audio tensor of shape
                (batch_size, num_channels, num_samples).
            input_lengths (torch.Tensor, optional): A tensor containing the lengths
                of the input sequences. If provided, it should have the shape
                (batch_size,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - log_amp (torch.Tensor): The computed log-amplitude spectrogram
                  of shape (batch_size, num_freq_bins, time_steps).
                - feats_lens (torch.Tensor): The lengths of the features for each
                  input in the batch.

        Raises:
            AssertionError: If the input STFT output does not have the expected
            dimensions or if the last dimension does not correspond to the real
            and imaginary parts.

        Examples:
            >>> log_spectrogram = LogSpectrogram()
            >>> input_tensor = torch.randn(2, 1, 16000)  # Example input tensor
            >>> output, lengths = log_spectrogram.forward(input_tensor)
            >>> print(output.shape)  # Output shape: (2, num_freq_bins, time_steps)

        Note:
            The definition of log-amplitude spectrogram differs between TTS and ASR:
            - TTS: log_10(abs(stft))
            - ASR: log_e(power(stft))
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

    def spectrogram(self) -> bool:
        """
            Conventional frontend structure for Automatic Speech Recognition (ASR).

        This class processes input audio signals through Short-Time Fourier Transform
        (STFT) to produce log-amplitude spectrograms, which are commonly used in
        speech processing tasks.

        The processing flow is as follows:
            1. Apply STFT to convert time-domain signal to time-frequency domain.
            2. Compute the log-amplitude of the power spectrum.

        Attributes:
            n_fft (int): The number of FFT points.
            hop_length (int): The number of samples between successive frames.
            win_length (Optional[int]): The length of each windowed signal segment.
            window (Optional[str]): The type of window function applied.
            stft (Stft): An instance of the Stft class used for STFT processing.

        Args:
            n_fft (int): Number of FFT points. Default is 1024.
            win_length (int, optional): Length of the window. Default is None.
            hop_length (int): Number of samples between frames. Default is 256.
            window (str, optional): Window type. Default is "hann".
            center (bool): Whether to pad the signal to center the window. Default is True.
            normalized (bool): Whether to normalize the output. Default is False.
            onesided (bool): Whether to return a one-sided spectrum. Default is True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the log-amplitude
            spectrogram and the lengths of the features.

        Raises:
            AssertionError: If the input STFT does not have the expected dimensions.

        Examples:
            >>> log_spectrogram = LogSpectrogram(n_fft=1024, hop_length=256)
            >>> audio_input = torch.randn(1, 16000)  # Example audio tensor
            >>> log_amp, feats_lens = log_spectrogram.forward(audio_input)

        Note:
            The log-amplitude is computed differently for Text-To-Speech (TTS) and ASR.
            For TTS, it is defined as log_10(abs(stft)), while for ASR, it is defined
            as log_e(power(stft)).

        Todo:
            - Consider adding support for additional window functions.
        """
        return True
