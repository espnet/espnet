"""Shared audio loading utilities for enhancement metrics."""

from pathlib import Path

import numpy as np
import resampy
import soundfile as sf


def load_audio(path: str | Path, sample_rate: int | None = None) -> np.ndarray:
    """Load mono audio and optionally resample it.

    Args:
        path: Path to a waveform file.
        sample_rate: Target sampling rate. If omitted, keep the original rate.

    Returns:
        A one-dimensional ``float32`` waveform.
    """
    waveform, current_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sample_rate is not None and current_rate != sample_rate:
        waveform = resampy.resample(waveform, current_rate, sample_rate)
    return np.asarray(waveform, dtype=np.float32)
