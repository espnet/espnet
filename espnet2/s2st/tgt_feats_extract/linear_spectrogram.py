from typing import Any, Dict, Optional, Tuple

import torch
from typeguard import typechecked

from espnet2.layers.stft import Stft
from espnet2.s2st.tgt_feats_extract.abs_tgt_feats_extract import AbsTgtFeatsExtract


class LinearSpectrogram(AbsTgtFeatsExtract):
    """
        Linear amplitude spectrogram extraction.

    This class implements the extraction of a linear amplitude spectrogram from audio
    signals. It utilizes the Short-Time Fourier Transform (STFT) to convert time-domain
    signals into the frequency domain and subsequently computes the amplitude spectrum.

    Attributes:
        n_fft (int): The number of FFT points.
        win_length (Optional[int]): The length of the windowed signal segments.
        hop_length (int): The number of samples between successive frames.
        window (Optional[str]): The window function to use (e.g., 'hann').
        stft (Stft): An instance of the STFT class for computing the Fourier transform.

    Args:
        n_fft (int): The number of FFT points (default is 1024).
        win_length (Optional[int]): The length of the windowed segments (default is None).
        hop_length (int): The number of samples between frames (default is 256).
        window (Optional[str]): The window function (default is 'hann').
        center (bool): If True, the signal is padded so that the window is centered
            at the current sample (default is True).
        normalized (bool): If True, the output is normalized (default is False).
        onesided (bool): If True, only the non-negative frequency terms are returned
            (default is True).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the amplitude spectrum
        and the lengths of the features.

    Yields:
        None

    Raises:
        AssertionError: If the input STFT tensor does not have the expected dimensions
            or shape.

    Examples:
        >>> spectrogram_extractor = LinearSpectrogram(n_fft=2048, hop_length=512)
        >>> audio_tensor = torch.randn(1, 16000)  # Example audio tensor
        >>> amp_spectrogram, lengths = spectrogram_extractor.forward(audio_tensor)

    Note:
        The output amplitude spectrum is computed from the STFT by taking the square root
        of the power spectrum, which is derived from the real and imaginary parts of the
        STFT.

    Todo:
        - Add additional window functions as options.
        - Implement further parameter validation.
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
            Calculate the output size of the linear spectrogram.

        The output size is determined by the number of FFT points used in the
        Short-Time Fourier Transform (STFT). Specifically, the output size
        corresponds to the number of frequency bins in the resulting amplitude
        spectrogram, which is computed as `n_fft // 2 + 1`.

        Returns:
            int: The number of frequency bins in the output amplitude spectrogram.

        Examples:
            >>> ls = LinearSpectrogram(n_fft=1024)
            >>> ls.output_size()
            513

        Note:
            The output size will change if the `n_fft` parameter is modified
            during the initialization of the LinearSpectrogram instance.
        """
        return self.n_fft // 2 + 1

    def get_parameters(self) -> Dict[str, Any]:
        """
                Return the parameters required by Vocoder.

        This method provides a dictionary containing the necessary parameters
        for the vocoder, which include the number of FFT points, the shift
        length, the window length, and the window type used during the STFT
        computation.

        Returns:
            Dict[str, Any]: A dictionary with the following keys:
                - n_fft (int): Number of FFT points.
                - n_shift (int): Hop length or the number of samples to shift
                  between consecutive frames.
                - win_length (Optional[int]): Length of the window applied to
                  each segment of audio.
                - window (Optional[str]): Type of window function applied.

        Examples:
            >>> spectrogram = LinearSpectrogram(n_fft=2048, hop_length=512)
            >>> parameters = spectrogram.get_parameters()
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
        Compute the linear amplitude spectrogram from input audio.

        This method applies the Short-Time Fourier Transform (STFT) to the
        input tensor, which represents audio data, and converts the complex
        STFT output into an amplitude spectrogram. The method ensures that
        the output tensor is in the expected shape and format.

        Args:
            input (torch.Tensor): A tensor containing the input audio data,
                typically with shape (batch_size, num_channels, num_samples).
            input_lengths (torch.Tensor, optional): A tensor containing the
                lengths of each input sequence, with shape (batch_size,).
                If provided, it helps to compute the correct output lengths
                for the amplitude spectrogram.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - torch.Tensor: The computed amplitude spectrogram with shape
                  (..., F), where F is the number of frequency bins.
                - torch.Tensor: The lengths of the features computed from the
                  input, with shape (batch_size,).

        Raises:
            AssertionError: If the computed STFT output does not have at least
            4 dimensions or if the last dimension of the STFT output does not
            equal 2 (for real and imaginary parts).

        Examples:
            >>> model = LinearSpectrogram()
            >>> input_audio = torch.randn(2, 1, 16000)  # (batch_size, channels, samples)
            >>> input_lengths = torch.tensor([16000, 12000])  # lengths of each input
            >>> amp_spectrogram, feats_lens = model.forward(input_audio, input_lengths)
            >>> amp_spectrogram.shape
            torch.Size([2, 513])  # Example shape for the output spectrogram

        Note:
            The input tensor should be in the appropriate shape, and the
            lengths tensor should match the batch size.
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

    def spectrogram(self) -> bool:
        """
            Linear amplitude spectrogram.

        This class computes the linear amplitude spectrogram from audio signals
        using Short-Time Fourier Transform (STFT). It extracts features suitable
        for tasks such as speech synthesis or audio analysis.

        The input audio is transformed into a time-frequency representation
        using STFT, which is then converted into an amplitude spectrogram.

        Attributes:
            n_fft (int): The size of the FFT window. Default is 1024.
            hop_length (int): The number of samples to skip between frames.
                Default is 256.
            win_length (Optional[int]): The size of the window. If None, it
                defaults to n_fft.
            window (Optional[str]): The type of window to use. Default is "hann".
            stft (Stft): An instance of the Stft class used to compute STFT.

        Args:
            n_fft (int): The size of the FFT window.
            win_length (Optional[int]): The size of the window.
            hop_length (int): The number of samples to skip between frames.
            window (Optional[str]): The type of window to use.
            center (bool): Whether to pad the signal on both sides so that
                the frame is centered at the current sample. Default is True.
            normalized (bool): Whether to normalize the output. Default is False.
            onesided (bool): If True, the output will only contain the positive
                frequencies. Default is True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The amplitude spectrogram and
            the lengths of the features.

        Examples:
            >>> spectrogram = LinearSpectrogram(n_fft=2048, hop_length=512)
            >>> audio_input = torch.randn(1, 16000)  # Example audio input
            >>> amp_spectrogram, lengths = spectrogram.forward(audio_input)

        Note:
            The output of the forward method is an amplitude spectrogram, which
            is computed from the power spectrum derived from the STFT.

        Raises:
            AssertionError: If the dimensions of the input STFT are not as
            expected.

        Todo:
            Consider adding options for different types of window functions and
            improving error handling for unsupported configurations.
        """
        return True
