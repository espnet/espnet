import copy
from typing import Optional, Tuple, Union

import humanfriendly
import numpy as np
import torch
from torch_complex.tensor import ComplexTensor
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.layers.log_mel import LogMel
from espnet2.layers.stft import Stft
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet.nets.pytorch_backend.frontends.frontend import Frontend


class DefaultFrontend(AbsFrontend):
    """
        Conventional frontend structure for ASR.

    This class implements a standard frontend processing pipeline for Automatic Speech Recognition (ASR),
    consisting of the following steps: STFT -> WPE -> MVDR-Beamformer -> Power-spectrum -> Log-Mel-Fbank.

    Attributes:
        stft (Stft): Short-time Fourier transform module.
        frontend (Frontend): Speech enhancement frontend module.
        logmel (LogMel): Log-Mel filterbank feature extraction module.
        n_mels (int): Number of Mel filterbank channels.
        frontend_type (str): Type of frontend, set to "default".
        hop_length (int): Hop length for STFT.
        apply_stft (bool): Flag to determine whether to apply STFT.

    Args:
        fs (Union[int, str]): Sampling frequency of the input audio. Defaults to 16000.
        n_fft (int): FFT size. Defaults to 512.
        win_length (Optional[int]): Window length for STFT. Defaults to None.
        hop_length (int): Hop length for STFT. Defaults to 128.
        window (Optional[str]): Window function type. Defaults to "hann".
        center (bool): Whether to pad the input on both sides. Defaults to True.
        normalized (bool): Whether to normalize the STFT. Defaults to False.
        onesided (bool): Whether to return only one-sided spectrum. Defaults to True.
        n_mels (int): Number of Mel filterbank channels. Defaults to 80.
        fmin (Optional[int]): Minimum frequency for Mel filters. Defaults to None.
        fmax (Optional[int]): Maximum frequency for Mel filters. Defaults to None.
        htk (bool): Whether to use HTK formula for Mel scale. Defaults to False.
        frontend_conf (Optional[dict]): Configuration for the Frontend module. Defaults to None.
        apply_stft (bool): Whether to apply STFT. Defaults to True.

    Note:
        This class inherits from AbsFrontend and implements the conventional frontend structure
        used in many ASR systems. It combines multiple processing steps to convert raw audio
        input into features suitable for acoustic modeling.

    Examples:
        >>> frontend = DefaultFrontend(fs=16000, n_fft=512, n_mels=80)
        >>> input_audio = torch.randn(1, 16000)
        >>> input_lengths = torch.tensor([16000])
        >>> features, feat_lengths = frontend(input_audio, input_lengths)
        >>> features.shape
        torch.Size([1, 126, 80])
    """

    @typechecked
    def __init__(
        self,
        fs: Union[int, str] = 16000,
        n_fft: int = 512,
        win_length: Optional[int] = None,
        hop_length: int = 128,
        window: Optional[str] = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        n_mels: int = 80,
        fmin: Optional[int] = None,
        fmax: Optional[int] = None,
        htk: bool = False,
        frontend_conf: Optional[dict] = get_default_kwargs(Frontend),
        apply_stft: bool = True,
    ):
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)

        # Deepcopy (In general, dict shouldn't be used as default arg)
        frontend_conf = copy.deepcopy(frontend_conf)
        self.hop_length = hop_length

        if apply_stft:
            self.stft = Stft(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                center=center,
                window=window,
                normalized=normalized,
                onesided=onesided,
            )
        else:
            self.stft = None
        self.apply_stft = apply_stft

        if frontend_conf is not None:
            self.frontend = Frontend(idim=n_fft // 2 + 1, **frontend_conf)
        else:
            self.frontend = None

        self.logmel = LogMel(
            fs=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
        )
        self.n_mels = n_mels
        self.frontend_type = "default"

    def output_size(self) -> int:
        """
                Returns the output size of the frontend.

        Returns:
            int: The number of Mel filterbank channels (n_mels) used in the frontend.

        Note:
            This method is used to determine the dimensionality of the feature vectors
            produced by the frontend, which is essential for configuring subsequent
            components in the ASR pipeline.

        Examples:
            >>> frontend = DefaultFrontend(n_mels=80)
            >>> frontend.output_size()
            80
        """
        return self.n_mels

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
                Perform frontend processing on the input audio.

        This method applies the complete frontend processing pipeline to the input audio,
        including STFT, optional speech enhancement, channel selection, power spectrum
        computation, and log-mel feature extraction.

        Args:
            input (torch.Tensor): Input audio tensor of shape (Batch, Time).
            input_lengths (torch.Tensor): Tensor of input audio lengths of shape (Batch,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - input_feats (torch.Tensor): Processed features of shape (Batch, Length, Dim).
                - feats_lens (torch.Tensor): Lengths of processed features of shape (Batch,).

        Note:
            The processing steps include:
            1. Domain conversion (e.g., STFT)
            2. Optional speech enhancement
            3. Channel selection for multi-channel input
            4. Power spectrum computation
            5. Log-mel feature extraction

            During training, a random channel is selected for multi-channel input.
            During inference, the first channel is used.

        Examples:
            >>> frontend = DefaultFrontend(fs=16000, n_fft=512, n_mels=80)
            >>> input_audio = torch.randn(2, 16000)  # 2 utterances of 1 second each
            >>> input_lengths = torch.tensor([16000, 12000])  # Second utterance is shorter
            >>> features, feat_lengths = frontend(input_audio, input_lengths)
            >>> features.shape
            torch.Size([2, 126, 80])
            >>> feat_lengths
            tensor([126,  95])
        """
        # 1. Domain-conversion: e.g. Stft: time -> time-freq
        if self.stft is not None:
            input_stft, feats_lens = self._compute_stft(input, input_lengths)
        else:
            input_stft = ComplexTensor(input[..., 0], input[..., 1])
            feats_lens = input_lengths
        # 2. [Option] Speech enhancement
        if self.frontend is not None:
            assert isinstance(input_stft, ComplexTensor), type(input_stft)
            # input_stft: (Batch, Length, [Channel], Freq)
            input_stft, _, mask = self.frontend(input_stft, feats_lens)

        # 3. [Multi channel case]: Select a channel
        if input_stft.dim() == 4:
            # h: (B, T, C, F) -> h: (B, T, F)
            if self.training:
                # Select 1ch randomly
                ch = np.random.randint(input_stft.size(2))
                input_stft = input_stft[:, :, ch, :]
            else:
                # Use the first channel
                input_stft = input_stft[:, :, 0, :]

        # 4. STFT -> Power spectrum
        # h: ComplexTensor(B, T, F) -> torch.Tensor(B, T, F)
        input_power = input_stft.real**2 + input_stft.imag**2

        # 5. Feature transform e.g. Stft -> Log-Mel-Fbank
        # input_power: (Batch, [Channel,] Length, Freq)
        #       -> input_feats: (Batch, Length, Dim)
        input_feats, _ = self.logmel(input_power, feats_lens)

        return input_feats, feats_lens

    def _compute_stft(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> torch.Tensor:
        input_stft, feats_lens = self.stft(input, input_lengths)

        assert input_stft.dim() >= 4, input_stft.shape
        # "2" refers to the real/imag parts of Complex
        assert input_stft.shape[-1] == 2, input_stft.shape

        # Change torch.Tensor to ComplexTensor
        # input_stft: (..., F, 2) -> (..., F)
        input_stft = ComplexTensor(input_stft[..., 0], input_stft[..., 1])
        return input_stft, feats_lens
