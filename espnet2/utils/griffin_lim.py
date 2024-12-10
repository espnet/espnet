#!/usr/bin/env python3

"""Griffin-Lim related modules."""

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
from functools import partial
from typing import Optional

import librosa
import numpy as np
import torch
from packaging.version import parse as V
from typeguard import typechecked

EPS = 1e-10


@typechecked
def logmel2linear(
    lmspc: np.ndarray,
    fs: int,
    n_fft: int,
    n_mels: int,
    fmin: Optional[int] = None,
    fmax: Optional[int] = None,
) -> np.ndarray:
    """
        Convert log Mel filterbank to linear spectrogram.

    This function transforms a log Mel spectrogram into a linear spectrogram
    using the inverse of the Mel filterbank. It computes the inverse Mel
    basis and applies it to the log Mel spectrogram.

    Args:
        lmspc (np.ndarray): Log Mel filterbank with shape (T, n_mels).
        fs (int): Sampling frequency.
        n_fft (int): The number of FFT points.
        n_mels (int): The number of Mel basis.
        fmin (Optional[int]): Minimum frequency to analyze. If None, defaults to 0.
        fmax (Optional[int]): Maximum frequency to analyze. If None, defaults to fs / 2.

    Returns:
        np.ndarray: Linear spectrogram with shape (T, n_fft // 2 + 1).

    Examples:
        >>> import numpy as np
        >>> lmspc = np.random.rand(100, 40)  # Example log Mel spectrogram
        >>> fs = 16000
        >>> n_fft = 2048
        >>> n_mels = 40
        >>> linear_spc = logmel2linear(lmspc, fs, n_fft, n_mels)
        >>> print(linear_spc.shape)  # Should be (100, 1025) for 2048 FFT points

    Note:
        The function uses the `librosa` library to create the Mel filterbank
        and compute its pseudoinverse.
    """
    assert lmspc.shape[1] == n_mels
    fmin = 0 if fmin is None else fmin
    fmax = fs / 2 if fmax is None else fmax
    mspc = np.power(10.0, lmspc)
    mel_basis = librosa.filters.mel(
        sr=fs, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
    )
    inv_mel_basis = np.linalg.pinv(mel_basis)
    return np.maximum(EPS, np.dot(inv_mel_basis, mspc.T).T)


@typechecked
def griffin_lim(
    spc: np.ndarray,
    n_fft: int,
    n_shift: int,
    win_length: Optional[int] = None,
    window: Optional[str] = "hann",
    n_iter: Optional[int] = 32,
) -> np.ndarray:
    """
        Convert linear spectrogram into waveform using Griffin-Lim.

    This function reconstructs a waveform from a linear spectrogram using the
    Griffin-Lim algorithm, which iteratively refines the phase information to
    recover the audio signal.

    Args:
        spc: Linear spectrogram (T, n_fft // 2 + 1).
        n_fft: The number of FFT points.
        n_shift: Shift size in points.
        win_length: Window length in points (optional).
        window: Window function type (optional, default is "hann").
        n_iter: The number of iterations (optional, default is 32).

    Returns:
        Reconstructed waveform (N,).

    Examples:
        # Example of using griffin_lim to convert a linear spectrogram back to audio
        import numpy as np
        spc = np.random.rand(100, 1025)  # Example linear spectrogram
        waveform = griffin_lim(spc, n_fft=2048, n_shift=512)

    Note:
        This function requires librosa library version 0.7.0 or higher for the
        optimized Griffin-Lim algorithm. If an older version is used, a warning
        will be logged, and a slower implementation will be used.

    Raises:
        ValueError: If the input spectrogram shape does not match the expected
        dimensions.
    """
    # assert the size of input linear spectrogram
    assert spc.shape[1] == n_fft // 2 + 1

    if V(librosa.__version__) >= V("0.7.0"):
        # use librosa's fast Grriffin-Lim algorithm
        spc = np.abs(spc.T)
        y = librosa.griffinlim(
            S=spc,
            n_iter=n_iter,
            hop_length=n_shift,
            win_length=win_length,
            window=window,
            center=True if spc.shape[1] > 1 else False,
        )
    else:
        # use slower version of Grriffin-Lim algorithm
        logging.warning(
            "librosa version is old. use slow version of Grriffin-Lim algorithm."
            "if you want to use fast Griffin-Lim, please update librosa via "
            "`source ./path.sh && pip install librosa==0.7.0`."
        )
        cspc = np.abs(spc).astype(np.complex).T
        angles = np.exp(2j * np.pi * np.random.rand(*cspc.shape))
        y = librosa.istft(cspc * angles, n_shift, win_length, window=window)
        for i in range(n_iter):
            angles = np.exp(
                1j
                * np.angle(librosa.stft(y, n_fft, n_shift, win_length, window=window))
            )
            y = librosa.istft(cspc * angles, n_shift, win_length, window=window)

    return y


# TODO(kan-bayashi): write as torch.nn.Module
class Spectrogram2Waveform(object):
    """
        Spectrogram to waveform conversion module.

    This class provides functionality to convert log Mel filterbanks or linear
    spectrograms into waveforms using the Griffin-Lim algorithm. It allows for
    flexible configuration of parameters such as the sampling frequency, number
    of FFT points, and windowing options.

    Attributes:
        fs: Sampling frequency.
        logmel2linear: Function to convert log Mel filterbank to linear
            spectrogram.
        griffin_lim: Function to perform Griffin-Lim reconstruction.
        params: Dictionary containing the parameters used for conversion.

    Args:
        n_fft: The number of FFT points.
        n_shift: Shift size in points.
        fs: Sampling frequency (optional).
        n_mels: The number of mel basis (optional).
        win_length: Window length in points (optional).
        window: Window function type (default: "hann").
        fmin: Minimum frequency to analyze (optional).
        fmax: Maximum frequency to analyze (optional).
        griffin_lim_iters: The number of iterations for Griffin-Lim (default: 8).

    Examples:
        >>> model = Spectrogram2Waveform(n_fft=2048, n_shift=512, fs=16000)
        >>> logmel = torch.rand(100, 80)  # Example log Mel filterbank
        >>> waveform = model(logmel)

    Note:
        This module is designed to work with PyTorch tensors and expects the
        input spectrogram to be in the shape (T_feats, n_mels) or
        (T_feats, n_fft // 2 + 1).

    Todo:
        Implement as a torch.nn.Module.
    """

    @typechecked
    def __init__(
        self,
        n_fft: int,
        n_shift: int,
        fs: Optional[int] = None,
        n_mels: Optional[int] = None,
        win_length: Optional[int] = None,
        window: Optional[str] = "hann",
        fmin: Optional[int] = None,
        fmax: Optional[int] = None,
        griffin_lim_iters: Optional[int] = 8,
    ):
        """Initialize module.

        Args:
            fs: Sampling frequency.
            n_fft: The number of FFT points.
            n_shift: Shift size in points.
            n_mels: The number of mel basis.
            win_length: Window length in points.
            window: Window function type.
            f_min: Minimum frequency to analyze.
            f_max: Maximum frequency to analyze.
            griffin_lim_iters: The number of iterations.

        """
        self.fs = fs
        self.logmel2linear = (
            partial(
                logmel2linear, fs=fs, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
            )
            if n_mels is not None
            else None
        )
        self.griffin_lim = partial(
            griffin_lim,
            n_fft=n_fft,
            n_shift=n_shift,
            win_length=win_length,
            window=window,
            n_iter=griffin_lim_iters,
        )
        self.params = dict(
            n_fft=n_fft,
            n_shift=n_shift,
            win_length=win_length,
            window=window,
            n_iter=griffin_lim_iters,
        )
        if n_mels is not None:
            self.params.update(fs=fs, n_mels=n_mels, fmin=fmin, fmax=fmax)

    def __repr__(self):
        retval = f"{self.__class__.__name__}("
        for k, v in self.params.items():
            retval += f"{k}={v}, "
        retval += ")"
        return retval

    def __call__(self, spc: torch.Tensor) -> torch.Tensor:
        """Convert spectrogram to waveform.

        Args:
            spc: Log Mel filterbank (T_feats, n_mels)
                or linear spectrogram (T_feats, n_fft // 2 + 1).

        Returns:
            Tensor: Reconstructed waveform (T_wav,).

        """
        device = spc.device
        dtype = spc.dtype
        spc = spc.cpu().numpy()
        if self.logmel2linear is not None:
            spc = self.logmel2linear(spc)
        wav = self.griffin_lim(spc)
        return torch.tensor(wav).to(device=device, dtype=dtype)
