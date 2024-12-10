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
    DefaultFrontend is a conventional frontend structure for automatic speech
    recognition (ASR). It processes audio signals through a series of transformations,
    including Short-Time Fourier Transform (STFT), Weighted Prediction Error (WPE),
    Minimum Variance Distortionless Response (MVDR) beamforming, power spectrum
    calculation, and finally converts to Log-Mel filterbanks.

    The processing flow is as follows:
    STFT -> WPE -> MVDR-Beamformer -> Power-spec -> Log-Mel-Fbank

    Attributes:
        hop_length (int): The number of audio samples between adjacent STFT frames.
        apply_stft (bool): Flag to indicate if STFT should be applied.
        frontend (Frontend): The frontend model for speech enhancement, if applied.
        logmel (LogMel): The Log-Mel filterbank layer.
        n_mels (int): Number of Mel frequency bins.

    Args:
        fs (Union[int, str]): Sampling frequency (default is 16000).
        n_fft (int): Number of FFT points (default is 512).
        win_length (Optional[int]): Length of the window (default is None).
        hop_length (int): Number of samples between frames (default is 128).
        window (Optional[str]): Window function (default is "hann").
        center (bool): If True, pads input such that the frame is centered at
            the original time index (default is True).
        normalized (bool): If True, normalize the output of STFT (default is False).
        onesided (bool): If True, returns only the positive frequency components
            (default is True).
        n_mels (int): Number of Mel bands to generate (default is 80).
        fmin (Optional[int]): Minimum frequency (default is None).
        fmax (Optional[int]): Maximum frequency (default is None).
        htk (bool): If True, use HTK formula for Mel filterbank (default is False).
        frontend_conf (Optional[dict]): Configuration for the frontend model (default
            is a copy of Frontend's default kwargs).
        apply_stft (bool): Flag to apply STFT (default is True).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The processed features and their lengths.

    Raises:
        AssertionError: If input dimensions do not match expected shapes.

    Examples:
        # Initialize the frontend
        frontend = DefaultFrontend(fs=16000, n_fft=512, n_mels=80)

        # Process an input tensor
        input_tensor = torch.randn(2, 16000)  # Batch of 2, 1 second audio
        input_lengths = torch.tensor([16000, 16000])  # Lengths of each input
        features, lengths = frontend(input_tensor, input_lengths)

    Note:
        Ensure that the input tensor has the correct shape and type before
        processing. The input should be a 2D tensor of shape (batch_size,
        num_samples).

    Todo:
        - Implement additional speech enhancement techniques as needed.
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
        Return the output size of the frontend.

        This method returns the number of Mel frequency bands that are
        produced by the frontend's log-mel layer. The output size is
        typically determined by the `n_mels` parameter set during
        initialization.

        Returns:
            int: The number of Mel frequency bands produced by the frontend.

        Examples:
            >>> frontend = DefaultFrontend(n_mels=40)
            >>> frontend.output_size()
            40

        Note:
            This method is particularly useful for determining the shape
            of the output tensor after feature extraction, especially
            in the context of downstream tasks such as ASR.
        """
        return self.n_mels

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the DefaultFrontend class.

        This method processes the input tensor through various stages of the
        frontend pipeline, which includes domain conversion via Short-Time
        Fourier Transform (STFT), optional speech enhancement, channel
        selection for multi-channel input, and transformation to Log-Mel
        features.

        Args:
            input (torch.Tensor): The input tensor containing audio waveforms.
                The expected shape is (Batch, Length) for single-channel audio
                or (Batch, Length, Channels) for multi-channel audio.
            input_lengths (torch.Tensor): A tensor of shape (Batch,) containing
                the lengths of each input sequence. This is used to handle
                variable-length inputs.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - torch.Tensor: The extracted features of shape
                  (Batch, Length, Dim), where Dim is the number of Mel bands.
                - torch.Tensor: The lengths of the extracted features.

        Raises:
            AssertionError: If the dimensions of the input STFT do not meet
            the expected requirements.

        Examples:
            >>> frontend = DefaultFrontend()
            >>> audio_input = torch.randn(2, 16000)  # Batch of 2, 1 second audio
            >>> input_lengths = torch.tensor([16000, 16000])  # Lengths of inputs
            >>> features, lengths = frontend.forward(audio_input, input_lengths)
            >>> print(features.shape)  # Should print: torch.Size([2, Length, 80])

        Note:
            The `apply_stft` argument in the constructor determines whether
            to apply STFT to the input. If set to False, the input should be
            a complex tensor.

        Todo:
            Consider adding support for more frontend processing options
            in future releases.
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
