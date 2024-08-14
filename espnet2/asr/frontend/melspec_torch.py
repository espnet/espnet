#!/usr/bin/env python3
#  2023, Jee-weon Jung, CMU
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Torchaudio MFCC"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio as ta
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend


class MelSpectrogramTorch(AbsFrontend):
    """
        Mel-Spectrogram using Torchaudio Implementation.

    This class implements a Mel-Spectrogram frontend using Torchaudio's MelSpectrogram
    transform. It can optionally apply pre-emphasis, logarithmic scaling, and
    normalization to the input audio signal.

    Attributes:
        log (bool): Whether to apply logarithmic scaling to the spectrogram.
        n_mels (int): Number of Mel filterbanks.
        preemp (bool): Whether to apply pre-emphasis to the input signal.
        normalize (Optional[str]): Normalization method ('mn' for mean normalization or None).
        window_fn (Callable): Window function (torch.hann_window or torch.hamming_window).
        flipped_filter (torch.Tensor): Pre-emphasis filter coefficients.
        transform (torchaudio.transforms.MelSpectrogram): Mel-spectrogram transform object.

    Note:
        This class inherits from AbsFrontend and is designed to be used as a frontend
        in speech processing tasks, particularly in the ESPnet2 framework.

    Example:
        >>> frontend = MelSpectrogramTorch(n_mels=80, n_fft=512, win_length=400, hop_length=160)
        >>> input_signal = torch.randn(1, 16000)
        >>> input_lengths = torch.tensor([16000])
        >>> mel_spec, output_lengths = frontend(input_signal, input_lengths)
    """

    @typechecked
    def __init__(
        self,
        preemp: bool = True,
        n_fft: int = 512,
        log: bool = False,
        win_length: int = 400,
        hop_length: int = 160,
        f_min: int = 20,
        f_max: int = 7600,
        n_mels: int = 80,
        window_fn: str = "hamming",
        mel_scale: str = "htk",
        normalize: Optional[str] = None,
    ):
        super().__init__()

        self.log = log
        self.n_mels = n_mels
        self.preemp = preemp
        self.normalize = normalize
        if window_fn == "hann":
            self.window_fn = torch.hann_window
        elif window_fn == "hamming":
            self.window_fn = torch.hamming_window

        if preemp:
            self.register_buffer(
                "flipped_filter",
                torch.FloatTensor([-0.97, 1.0]).unsqueeze(0).unsqueeze(0),
            )

        self.transform = ta.transforms.MelSpectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            window_fn=self.window_fn,
            mel_scale=mel_scale,
        )

    def forward(
        self, input: torch.Tensor, input_length: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
                Compute the Mel-spectrogram of the input audio signal.

        This method applies the Mel-spectrogram transformation to the input audio signal,
        optionally performing pre-emphasis, logarithmic scaling, and normalization.

        Args:
            input (torch.Tensor): Input audio signal tensor of shape (batch_size, num_samples).
            input_length (torch.Tensor): Tensor containing the lengths of each audio signal
                in the batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - torch.Tensor: Mel-spectrogram features of shape (batch_size, num_frames, n_mels).
                - torch.Tensor: Output lengths tensor containing the number of frames for each
                  spectrogram in the batch.

        Raises:
            AssertionError: If the input tensor does not have exactly 2 dimensions.

        Note:
            This method uses torch.no_grad() and torch.cuda.amp.autocast(enabled=False)
            to ensure consistent behavior and avoid unnecessary gradient computations.

        Example:
            >>> frontend = MelSpectrogramTorch(n_mels=80, n_fft=512, win_length=400, hop_length=160)
            >>> input_signal = torch.randn(2, 16000)  # Batch of 2 audio signals
            >>> input_lengths = torch.tensor([16000, 16000])
            >>> mel_spec, output_lengths = frontend(input_signal, input_lengths)
            >>> print(mel_spec.shape)  # Expected output: torch.Size([2, num_frames, 80])
        """
        # input check
        assert (
            len(input.size()) == 2
        ), "The number of dimensions of input tensor must be 2!"

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                if self.preemp:
                    # reflect padding to match lengths of in/out
                    x = input.unsqueeze(1)
                    x = F.pad(x, (1, 0), "reflect")

                    # apply preemphasis
                    x = F.conv1d(x, self.flipped_filter).squeeze(1)
                else:
                    x = input

                # apply frame feature extraction
                x = self.transform(x)

                if self.log:
                    x = torch.log(x + 1e-6)
                if self.normalize is not None:
                    if self.normalize == "mn":
                        x = x - torch.mean(x, dim=-1, keepdim=True)
                    else:
                        raise NotImplementedError(
                            f"got {self.normalize}, not implemented"
                        )

        input_length = torch.Tensor([x.size(-1)]).repeat(x.size(0))

        return x.permute(0, 2, 1), input_length

    def output_size(self) -> int:
        """
                Return the output size of the Mel-spectrogram feature dimension.

        This method returns the number of Mel filterbanks used in the Mel-spectrogram
        computation, which corresponds to the size of the feature dimension in the
        output.

        Returns:
            int: The number of Mel filterbanks (n_mels) used in the Mel-spectrogram
            computation.

        Example:
            >>> frontend = MelSpectrogramTorch(n_mels=80)
            >>> feature_dim = frontend.output_size()
            >>> print(feature_dim)  # Expected output: 80
        """
        return self.n_mels
