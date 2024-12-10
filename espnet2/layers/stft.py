from typing import Optional, Tuple, Union

import librosa
import numpy as np
import torch
from packaging.version import parse as V
from torch_complex.tensor import ComplexTensor
from typeguard import typechecked

from espnet2.enh.layers.complex_utils import to_complex
from espnet2.layers.inversible_interface import InversibleInterface
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

is_torch_1_10_plus = V(torch.__version__) >= V("1.10.0")


class Stft(torch.nn.Module, InversibleInterface):
    """
        Stft is a PyTorch module for computing the Short-Time Fourier Transform (STFT)
    and its inverse. It provides an efficient implementation for transforming time-domain
    signals into the frequency domain, supporting multi-channel inputs.

    Attributes:
        n_fft (int): Number of FFT points.
        win_length (int): Length of the window. If None, defaults to n_fft.
        hop_length (int): Number of samples between each STFT frame.
        window (str): Type of window to apply. Must be a valid PyTorch window function.
        center (bool): If True, pads the input signal to ensure that the frames are centered.
        normalized (bool): If True, normalizes the output by the window length.
        onesided (bool): If True, returns a one-sided spectrum.

    Args:
        n_fft (int): Number of FFT points (default: 512).
        win_length (Optional[int]): Length of the window (default: None).
        hop_length (int): Number of samples between frames (default: 128).
        window (Optional[str]): Type of window (default: "hann").
        center (bool): Center the signal before processing (default: True).
        normalized (bool): Normalize the output (default: False).
        onesided (bool): Return one-sided spectrum (default: True).

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]:
            output: The STFT output tensor of shape
                    (Batch, Frames, Freq, 2) or
                    (Batch, Frames, Channels, Freq, 2).
            ilens: Optional tensor indicating the lengths of the input signals.

    Yields:
        None

    Raises:
        ValueError: If the specified window is not implemented in PyTorch.
        NotImplementedError: If called in training mode on devices not supporting
        the training mode with librosa.

    Examples:
        # Create an instance of Stft
        stft = Stft(n_fft=1024, hop_length=256)

        # Compute the STFT
        input_tensor = torch.randn(8, 16000)  # Batch of 8 audio samples
        output, ilens = stft(input_tensor)

        # Compute the inverse STFT
        reconstructed_wavs, ilens = stft.inverse(output, ilens)

    Note:
        The STFT implementation is compatible with librosa's STFT regarding
        padding and scaling. Note that it differs from scipy.signal.stft.

    Todo:
        - Add support for additional window types.
        - Implement further optimizations for the inverse STFT process.
    """

    @typechecked
    def __init__(
        self,
        n_fft: int = 512,
        win_length: Optional[int] = None,
        hop_length: int = 128,
        window: Optional[str] = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
    ):
        super().__init__()
        self.n_fft = n_fft
        if win_length is None:
            self.win_length = n_fft
        else:
            self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        if window is not None and not hasattr(torch, f"{window}_window"):
            raise ValueError(f"{window} window is not implemented")
        self.window = window

    def extra_repr(self):
        """
            Returns a string representation of the STFT parameters for logging.

        This method provides a summary of the key parameters used in the STFT
        (Short-Time Fourier Transform) configuration. The output is useful for
        debugging and understanding the current setup of the STFT instance.

        Attributes:
            n_fft (int): The number of FFT points.
            win_length (int): The length of each windowed segment.
            hop_length (int): The number of samples between successive frames.
            center (bool): Whether to pad the input signal on both sides.
            normalized (bool): Whether to normalize the output.
            onesided (bool): Whether to return a one-sided spectrum.

        Returns:
            str: A string representation of the STFT parameters.

        Examples:
            >>> stft = Stft(n_fft=1024, win_length=512, hop_length=256)
            >>> print(stft.extra_repr())
            n_fft=1024, win_length=512, hop_length=256, center=True,
            normalized=False, onesided=True
        """
        return (
            f"n_fft={self.n_fft}, "
            f"win_length={self.win_length}, "
            f"hop_length={self.hop_length}, "
            f"center={self.center}, "
            f"normalized={self.normalized}, "
            f"onesided={self.onesided}"
        )

    def forward(
        self, input: torch.Tensor, ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        STFT forward function.

        Computes the Short-Time Fourier Transform (STFT) of the input tensor.

        Args:
            input: A tensor of shape (Batch, Nsamples) or
                   (Batch, Nsample, Channels) representing the audio signal.
            ilens: An optional tensor of shape (Batch) that specifies the length
                   of each input signal. If provided, it will be used to mask
                   the output.

        Returns:
            output: A tuple containing:
                - A tensor of shape (Batch, Frames, Freq, 2) or
                  (Batch, Frames, Channels, Freq, 2) representing the STFT
                  output in the format of real and imaginary components.
                - An optional tensor of shape (Batch) that contains the
                  lengths of the output signals after STFT.

        Note:
            The output tensor contains the STFT results with real and
            imaginary parts represented in the last dimension. The input
            tensor can be either a single channel or multi-channel audio
            signal.

        Examples:
            >>> stft_layer = Stft(n_fft=512, hop_length=128)
            >>> audio_input = torch.randn(10, 16000)  # 10 samples, 16000 audio length
            >>> output, output_lengths = stft_layer(audio_input)
        """
        bs = input.size(0)
        if input.dim() == 3:
            multi_channel = True
            # input: (Batch, Nsample, Channels) -> (Batch * Channels, Nsample)
            input = input.transpose(1, 2).reshape(-1, input.size(1))
        else:
            multi_channel = False

        # NOTE(kamo):
        #   The default behaviour of torch.stft is compatible with librosa.stft
        #   about padding and scaling.
        #   Note that it's different from scipy.signal.stft

        # output: (Batch, Freq, Frames, 2=real_imag)
        # or (Batch, Channel, Freq, Frames, 2=real_imag)
        if self.window is not None:
            window_func = getattr(torch, f"{self.window}_window")
            window = window_func(
                self.win_length, dtype=input.dtype, device=input.device
            )
        else:
            window = None

        # For the compatibility of ARM devices, which do not support
        # torch.stft() due to the lack of MKL (on older pytorch versions),
        # there is an alternative replacement implementation with librosa.
        # Note: pytorch >= 1.10.0 now has native support for FFT and STFT
        # on all cpu targets including ARM.
        if input.is_cuda or torch.backends.mkl.is_available() or is_torch_1_10_plus:
            stft_kwargs = dict(
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                center=self.center,
                window=window,
                normalized=self.normalized,
                onesided=self.onesided,
            )
            stft_kwargs["return_complex"] = True
            # NOTE(Jinchuan) CuFFT is not compatible with bfloat16
            output = torch.stft(input.float(), **stft_kwargs)
            output = torch.view_as_real(output).type(input.dtype)
        else:
            if self.training:
                raise NotImplementedError(
                    "stft is implemented with librosa on this device, which does not "
                    "support the training mode."
                )

            # use stft_kwargs to flexibly control different PyTorch versions' kwargs
            # note: librosa does not support a win_length that is < n_ftt
            # but the window can be manually padded (see below).
            stft_kwargs = dict(
                n_fft=self.n_fft,
                win_length=self.n_fft,
                hop_length=self.hop_length,
                center=self.center,
                window=window,
                pad_mode="reflect",
            )

            if window is not None:
                # pad the given window to n_fft
                n_pad_left = (self.n_fft - window.shape[0]) // 2
                n_pad_right = self.n_fft - window.shape[0] - n_pad_left
                stft_kwargs["window"] = torch.cat(
                    [torch.zeros(n_pad_left), window, torch.zeros(n_pad_right)], 0
                ).numpy()
            else:
                win_length = (
                    self.win_length if self.win_length is not None else self.n_fft
                )
                stft_kwargs["window"] = torch.ones(win_length)

            output = []
            # iterate over istances in a batch
            for i, instance in enumerate(input):
                stft = librosa.stft(input[i].numpy(), **stft_kwargs)
                output.append(torch.tensor(np.stack([stft.real, stft.imag], -1)))
            output = torch.stack(output, 0)
            if not self.onesided:
                len_conj = self.n_fft - output.shape[1]
                conj = output[:, 1 : 1 + len_conj].flip(1)
                conj[:, :, :, -1].data *= -1
                output = torch.cat([output, conj], 1)
            if self.normalized:
                output = output * (stft_kwargs["window"].shape[0] ** (-0.5))

        # output: (Batch, Freq, Frames, 2=real_imag)
        # -> (Batch, Frames, Freq, 2=real_imag)
        output = output.transpose(1, 2)
        if multi_channel:
            # output: (Batch * Channel, Frames, Freq, 2=real_imag)
            # -> (Batch, Frame, Channel, Freq, 2=real_imag)
            output = output.view(bs, -1, output.size(1), output.size(2), 2).transpose(
                1, 2
            )

        if ilens is not None:
            if self.center:
                pad = self.n_fft // 2
                ilens = ilens + 2 * pad

            olens = (
                torch.div(ilens - self.n_fft, self.hop_length, rounding_mode="trunc")
                + 1
            )
            output.masked_fill_(make_pad_mask(olens, output, 1), 0.0)
        else:
            olens = None

        return output, olens

    def inverse(
        self, input: Union[torch.Tensor, ComplexTensor], ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Inverse STFT.

        This function computes the inverse Short-Time Fourier Transform (iSTFT)
        of the given input tensor, which can be a standard tensor or a complex
        tensor. The inverse STFT is used to reconstruct the time-domain signal
        from its frequency-domain representation.

        Args:
            input: A tensor of shape (batch, T, F, 2) representing the complex
                STFT output, or a ComplexTensor of shape (batch, T, F).
            ilens: A tensor of shape (batch,) containing the lengths of the
                original signals. If provided, it will be used to set the
                output lengths accordingly.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - wavs: A tensor of shape (batch, samples) containing the
                  reconstructed time-domain waveforms.
                - ilens: A tensor of shape (batch,) containing the lengths
                  of the reconstructed signals.

        Examples:
            >>> stft_layer = Stft()
            >>> input_tensor = torch.randn(2, 100, 64, 2)  # Example STFT output
            >>> lengths = torch.tensor([100, 80])  # Example input lengths
            >>> reconstructed_wavs, output_lengths = stft_layer.inverse(input_tensor, lengths)
        """
        input = to_complex(input)

        if self.window is not None:
            window_func = getattr(torch, f"{self.window}_window")
            datatype = input.real.dtype
            window = window_func(self.win_length, dtype=datatype, device=input.device)
        else:
            window = None

        input = input.transpose(1, 2)

        wavs = torch.functional.istft(
            input,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            normalized=self.normalized,
            onesided=self.onesided,
            length=ilens.max() if ilens is not None else ilens,
            return_complex=False,
        )

        return wavs, ilens
