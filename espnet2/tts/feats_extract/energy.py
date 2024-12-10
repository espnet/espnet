# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Energy extractor."""

from typing import Any, Dict, Optional, Tuple, Union

import humanfriendly
import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.layers.stft import Stft
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract
from espnet.nets.pytorch_backend.nets_utils import pad_list


class Energy(AbsFeatsExtract):
    """
        Energy extractor for audio features.

    This class implements an energy extraction mechanism from audio signals. It
    utilizes Short-Time Fourier Transform (STFT) to compute the energy of the
    input audio and offers functionalities to adjust and average the energy
    based on token durations.

    Attributes:
        fs (int): Sampling frequency of the input audio.
        n_fft (int): Number of FFT points.
        win_length (Optional[int]): Length of the window for STFT.
        hop_length (int): Hop length for STFT.
        window (str): Type of window to use for STFT.
        center (bool): Whether to center the input signal.
        normalized (bool): Whether to normalize the output.
        onesided (bool): Whether to use one-sided STFT.
        use_token_averaged_energy (bool): Whether to use averaged energy per token.
        reduction_factor (Optional[int]): Factor for reducing the energy length.

    Args:
        fs (Union[int, str]): Sampling frequency of the audio. Can be an int or
            a human-friendly string (e.g., "22k").
        n_fft (int): Number of FFT points. Default is 1024.
        win_length (Optional[int]): Length of the window. Default is None, which
            uses `n_fft`.
        hop_length (int): Hop length for STFT. Default is 256.
        window (str): Type of window function to use. Default is "hann".
        center (bool): Whether to center the input signal. Default is True.
        normalized (bool): Whether to normalize the output. Default is False.
        onesided (bool): Whether to use one-sided STFT. Default is True.
        use_token_averaged_energy (bool): Whether to average energy per token.
            Default is True.
        reduction_factor (Optional[int]): Factor for reducing the energy length.
            Must be >= 1 if `use_token_averaged_energy` is True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - energy (torch.Tensor): Extracted energy of shape (B, T, 1).
            - energy_lengths (torch.Tensor): Lengths of the energy sequences.

    Raises:
        AssertionError: If `reduction_factor` is less than 1 when
            `use_token_averaged_energy` is True.

    Examples:
        >>> energy_extractor = Energy(fs=22050, n_fft=1024, hop_length=256)
        >>> input_audio = torch.randn(10, 16000)  # Batch of 10 audio signals
        >>> energy, lengths = energy_extractor(input_audio)

    Note:
        This class is a subclass of `AbsFeatsExtract` and must be used
        in accordance with its interface.
    """

    @typechecked
    def __init__(
        self,
        fs: Union[int, str] = 22050,
        n_fft: int = 1024,
        win_length: Optional[int] = None,
        hop_length: int = 256,
        window: str = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        use_token_averaged_energy: bool = True,
        reduction_factor: Optional[int] = None,
    ):
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)

        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.use_token_averaged_energy = use_token_averaged_energy
        if use_token_averaged_energy:
            assert reduction_factor >= 1
        self.reduction_factor = reduction_factor

        self.stft = Stft(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=window,
            center=center,
            normalized=normalized,
            onesided=onesided,
        )

    def output_size(self) -> int:
        """
            Returns the output size of the energy extractor.

        This method returns a fixed output size of 1, which corresponds to the
        energy feature extracted from the input signal. This is a property of the
        energy extraction process, as the energy feature is a scalar value for
        each input frame.

        Returns:
            int: The output size of the energy extractor, which is always 1.

        Examples:
            >>> energy_extractor = Energy()
            >>> output_size = energy_extractor.output_size()
            >>> print(output_size)
            1
        """
        return 1

    def get_parameters(self) -> Dict[str, Any]:
        """
        Retrieve the parameters of the Energy extractor.

        This method returns a dictionary containing the key parameters
        used in the Energy extractor, which are essential for understanding
        the configuration and setup of the feature extraction process.

        Returns:
            Dict[str, Any]: A dictionary containing the parameters of the
            Energy extractor, including sample rate, FFT size, hop length,
            window type, and other relevant configurations.

        Examples:
            >>> energy_extractor = Energy()
            >>> parameters = energy_extractor.get_parameters()
            >>> print(parameters)
            {
                'fs': 22050,
                'n_fft': 1024,
                'hop_length': 256,
                'window': 'hann',
                'win_length': None,
                'center': True,
                'normalized': False,
                'use_token_averaged_energy': True,
                'reduction_factor': None
            }
        """
        return dict(
            fs=self.fs,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            win_length=self.win_length,
            center=self.stft.center,
            normalized=self.stft.normalized,
            use_token_averaged_energy=self.use_token_averaged_energy,
            reduction_factor=self.reduction_factor,
        )

    def forward(
        self,
        input: torch.Tensor,
        input_lengths: torch.Tensor = None,
        feats_lengths: torch.Tensor = None,
        durations: torch.Tensor = None,
        durations_lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Extracts energy features from audio input.

        This class inherits from the abstract class `AbsFeatsExtract` and provides
        methods to compute energy features from audio signals using Short-Time
        Fourier Transform (STFT). It supports various configurations for the STFT
        and energy calculation.

        Attributes:
            fs (int): Sampling frequency of the audio signal.
            n_fft (int): Number of FFT points.
            hop_length (int): Number of samples between successive frames.
            win_length (Optional[int]): Length of each windowed segment.
            window (str): Type of window function to use (e.g., "hann").
            use_token_averaged_energy (bool): Whether to use token-averaged energy.
            reduction_factor (Optional[int]): Factor by which to reduce energy.

        Args:
            fs (Union[int, str]): Sampling frequency, can be an integer or a string.
            n_fft (int): Number of FFT points (default: 1024).
            win_length (Optional[int]): Length of each windowed segment (default: None).
            hop_length (int): Number of samples between successive frames (default: 256).
            window (str): Type of window function to use (default: "hann").
            center (bool): Whether to center the signal (default: True).
            normalized (bool): Whether to normalize the output (default: False).
            onesided (bool): Whether to use one-sided spectrum (default: True).
            use_token_averaged_energy (bool): Whether to use token-averaged energy
                (default: True).
            reduction_factor (Optional[int]): Factor by which to reduce energy
                (default: None).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Energy features of shape (B, T, 1).
                - Lengths of the energy features.

        Raises:
            AssertionError: If the input tensor's shape is not valid or if the
                `reduction_factor` is invalid when `use_token_averaged_energy` is True.

        Examples:
            >>> energy_extractor = Energy(fs=22050, n_fft=1024)
            >>> input_tensor = torch.randn(10, 16000)  # Batch of 10 audio signals
            >>> energy, lengths = energy_extractor.forward(input_tensor)

        Note:
            The input tensor should have the shape (B, T), where B is the batch
            size and T is the number of time steps.
        """
        # If not provide, we assume that the inputs have the same length
        if input_lengths is None:
            input_lengths = (
                input.new_ones(input.shape[0], dtype=torch.long) * input.shape[1]
            )

        # Domain-conversion: e.g. Stft: time -> time-freq
        input_stft, energy_lengths = self.stft(input, input_lengths)

        assert input_stft.dim() >= 4, input_stft.shape
        assert input_stft.shape[-1] == 2, input_stft.shape

        # input_stft: (..., F, 2) -> (..., F)
        input_power = input_stft[..., 0] ** 2 + input_stft[..., 1] ** 2
        # sum over frequency (B, N, F) -> (B, N)
        energy = torch.sqrt(torch.clamp(input_power.sum(dim=2), min=1.0e-10))

        # (Optional): Adjust length to match with the mel-spectrogram
        if feats_lengths is not None:
            energy = [
                self._adjust_num_frames(e[:el].view(-1), fl)
                for e, el, fl in zip(energy, energy_lengths, feats_lengths)
            ]
            energy_lengths = feats_lengths

        # (Optional): Average by duration to calculate token-wise energy
        if self.use_token_averaged_energy:
            durations = durations * self.reduction_factor
            energy = [
                self._average_by_duration(e[:el].view(-1), d)
                for e, el, d in zip(energy, energy_lengths, durations)
            ]
            energy_lengths = durations_lengths

        # Padding
        if isinstance(energy, list):
            energy = pad_list(energy, 0.0)

        # Return with the shape (B, T, 1)
        return energy.unsqueeze(-1), energy_lengths

    def _average_by_duration(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        assert 0 <= len(x) - d.sum() < self.reduction_factor
        d_cumsum = F.pad(d.cumsum(dim=0), (1, 0))
        x_avg = [
            x[start:end].mean() if len(x[start:end]) != 0 else x.new_tensor(0.0)
            for start, end in zip(d_cumsum[:-1], d_cumsum[1:])
        ]
        return torch.stack(x_avg)

    @staticmethod
    def _adjust_num_frames(x: torch.Tensor, num_frames: torch.Tensor) -> torch.Tensor:
        if num_frames > len(x):
            x = F.pad(x, (0, num_frames - len(x)))
        elif num_frames < len(x):
            x = x[:num_frames]
        return x
