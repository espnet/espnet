#!/usr/bin/env python3

"""Griffin-Lim related modules."""

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

from distutils.version import LooseVersion
from typing import Optional
from typing import Union

import librosa
import numpy as np


EPS = 1e-10


def logmel2linear(
    lmspc: np.ndarray,
    fs: int,
    n_fft: int,
    n_mels: int,
    fmin: Optional[Union[int, None]] = None,
    fmax: Optional[Union[int, None]] = None,
) -> np.ndarray:
    """Convert log Mel filterbank to linear spectrogram.

    Args:
        lmspc: Log Mel filterbank (T, n_mels).
        fs: Sampling frequency.
        n_fft: The number of FFT points.
        n_mels: The number of mel basis.
        f_min: Minimum frequency to analyze.
        f_max: Maximum frequency to analyze.

    Returns:
        Linear spectrogram (T, n_fft // 2 + 1).

    """
    assert lmspc.shape[1] == n_mels
    fmin = 0 if fmin is None else fmin
    fmax = fs / 2 if fmax is None else fmax
    mspc = np.power(10.0, lmspc)
    mel_basis = librosa.filters.mel(fs, n_fft, n_mels, fmin, fmax)
    inv_mel_basis = np.linalg.pinv(mel_basis)
    spc = np.maximum(EPS, np.dot(inv_mel_basis, mspc.T).T)

    return spc


def griffin_lim(
    spc: np.ndarray,
    n_fft: int,
    n_shift: int,
    win_length: Optional[Union[int, None]] = None,
    window: Optional[str] = 'hann',
    num_iterations: Optional[int] = 32,
) -> np.ndarray:
    """Convert linear spectrogram into waveform using Griffin-Lim.

    Args:
        spc: Linear spectrogram (T, n_fft // 2 + 1).
        n_fft: The number of FFT points.
        n_shift: Shift size in points.
        win_length: Window length in points.
        window: Window function type.
        num_iterations: The number of iterations.

    Returns:
        Reconstructed waveform (N,).

    """
    # assert the size of input linear spectrogram
    assert spc.shape[1] == n_fft // 2 + 1

    if LooseVersion(librosa.__version__) >= LooseVersion('0.7.0'):
        # use librosa's fast Grriffin-Lim algorithm
        spc = np.abs(spc.T)
        y = librosa.griffinlim(
            S=spc,
            n_iter=num_iterations,
            hop_length=n_shift,
            win_length=win_length,
            window=window,
            center=True if spc.shape[1] > 1 else False
        )
    else:
        # use slower version of Grriffin-Lim algorithm
        logging.warning("librosa version is old. use slow version of Grriffin-Lim algorithm."
                        "if you want to use fast Griffin-Lim, please update librosa via "
                        "`source ./path.sh && pip install librosa==0.7.0`.")
        cspc = np.abs(spc).astype(np.complex).T
        angles = np.exp(2j * np.pi * np.random.rand(*cspc.shape))
        y = librosa.istft(cspc * angles, n_shift, win_length, window=window)
        for i in range(num_iterations):
            angles = np.exp(1j * np.angle(librosa.stft(y, n_fft, n_shift, win_length, window=window)))
            y = librosa.istft(cspc * angles, n_shift, win_length, window=window)

    return y


def spectrogram2wav(
    spc: np.ndarray,
    fs: int,
    n_fft: int,
    n_shift: int,
    n_mels: Union[int, None] = 80,
    win_length: Optional[Union[int, None]] = None,
    window: Optional[str] = 'hann',
    fmin: Optional[Union[int, None]] = None,
    fmax: Optional[Union[int, None]] = None,
    num_iterations: Optional[int] = 32,
) -> np.ndarray:
    """Convert spectrogram to waveform.

    Args:
        spc: Log Mel filterbank (T, n_mels) or linear spectrogram (T, n_fft // 2 + 1).
        fs: Sampling frequency.
        n_fft: The number of FFT points.
        n_shift: Shift size in points.
        n_mels: The number of mel basis.
        win_length: Window length in points.
        window: Window function type.
        f_min: Minimum frequency to analyze.
        f_max: Maximum frequency to analyze.
        num_iterations: The number of iterations.

    Returns:
        Reconstructed waveform (N,).

    """
    if n_mels is None:
        assert n_fft // 2 + 1 == spc.shape[1]
    else:
        assert n_mels == spc.shape[1]
        spc = logmel2linear(spc, fs, n_fft, n_mels, fmin, fmax)
    return griffin_lim(spc, n_fft, n_shift, win_length, window, num_iterations)
