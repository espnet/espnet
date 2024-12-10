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
    MelSpectrogramTorch is a class that computes the Mel-spectrogram of audio
    signals using the Torchaudio library. This class is part of the ESPnet2
    framework and extends the abstract frontend class AbsFrontend. It provides
    functionality to preprocess audio data into Mel-spectrograms, which are
    commonly used in speech recognition tasks.

    Attributes:
        log (bool): Indicates whether to apply logarithmic scaling to the output.
        n_mels (int): The number of Mel frequency bins.
        preemp (bool): Indicates whether to apply pre-emphasis to the input signal.
        normalize (Optional[str]): Method of normalization. Options include
            "mn" for mean normalization.
        window_fn (Callable): The window function to apply (Hanning or Hamming).

    Args:
        preemp (bool): Whether to apply pre-emphasis (default: True).
        n_fft (int): Number of FFT points (default: 512).
        log (bool): Whether to apply logarithmic scaling (default: False).
        win_length (int): Window length for FFT (default: 400).
        hop_length (int): Hop length for FFT (default: 160).
        f_min (int): Minimum frequency (default: 20).
        f_max (int): Maximum frequency (default: 7600).
        n_mels (int): Number of Mel bands (default: 80).
        window_fn (str): Type of window function ("hamming" or "hann", default: "hamming").
        mel_scale (str): Type of Mel scale ("htk" or other) (default: "htk").
        normalize (Optional[str]): Normalization method (default: None).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the Mel-spectrogram
        tensor and the tensor of input lengths.

    Raises:
        AssertionError: If the input tensor does not have 2 dimensions.
        NotImplementedError: If an unsupported normalization method is specified.

    Examples:
        >>> mel_spectrogram = MelSpectrogramTorch()
        >>> audio_tensor = torch.randn(1, 16000)  # Example audio tensor
        >>> input_length = torch.tensor([16000])   # Length of the input audio
        >>> mel_spec, mel_length = mel_spectrogram(audio_tensor, input_length)

    Note:
        This implementation utilizes GPU acceleration if available. Ensure that
        the input tensor is on the correct device.
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
        Compute the Mel-spectrogram of the input audio tensor.

        This method applies a series of transformations to the input tensor,
        including optional pre-emphasis, Mel-spectrogram conversion, and
        logarithmic scaling, to produce a time-frequency representation.

        Args:
            input (torch.Tensor): A 2D tensor of shape (batch_size, num_samples)
                representing the input audio waveform.
            input_length (torch.Tensor): A 1D tensor of shape (batch_size,)
                containing the lengths of the input audio samples.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - A 3D tensor of shape (batch_size, n_mels, num_frames)
                  representing the Mel-spectrogram features.
                - A 1D tensor of shape (batch_size,) containing the lengths
                  of the Mel-spectrogram features.

        Raises:
            AssertionError: If the input tensor does not have exactly 2 dimensions.

        Examples:
            >>> model = MelSpectrogramTorch()
            >>> audio_input = torch.randn(2, 16000)  # batch of 2 audio signals
            >>> input_length = torch.tensor([16000, 16000])  # lengths of audio
            >>> mel_spectrogram, mel_length = model(audio_input, input_length)
            >>> print(mel_spectrogram.shape)  # Output shape: (2, 80, num_frames)

        Note:
            The pre-emphasis step can be enabled or disabled via the constructor
            parameter `preemp`. The logarithmic scaling can be controlled with
            the `log` parameter.
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
        Returns the number of Mel frequency cepstral coefficients (MFCCs) generated
        by the MelSpectrogramTorch instance.

        This method provides the output size, which corresponds to the number of
        Mel bands specified during the initialization of the MelSpectrogramTorch
        class. It can be useful for understanding the shape of the output tensor
        produced by the forward method.

        Returns:
            int: The number of Mel bands (n_mels) used in the spectrogram.

        Examples:
            mel_spectrogram = MelSpectrogramTorch(n_mels=80)
            output_length = mel_spectrogram.output_size()
            print(output_length)  # Output: 80
        """
        return self.n_mels
