"""Shared waveform I/O helpers for enhancement metrics.

Metric classes under this package (`SISNRMetric`, `PESQMetric`, `STOIMetric`)
call ``load_audio`` to read reference/hypothesis paths from SCP rows. Keeping
the loader here avoids duplicating soundfile/resample logic in each metric.
"""

from pathlib import Path

import numpy as np
import resampy
import soundfile as sf


def load_audio(path: str | Path, sample_rate: int | None = None) -> np.ndarray:
    """Load mono audio and optionally resample it.

    Multi-channel files use the first (reference) channel rather than averaging.

    Args:
        path: Path to a waveform file.
        sample_rate: Target sampling rate. If omitted, keep the original rate.

    Returns:
        A one-dimensional ``float32`` waveform.
    """
    waveform, current_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        # Prefer the reference (first) channel over averaging across channels.
        waveform = waveform[:, 0]
    if sample_rate is not None and current_rate != sample_rate:
        waveform = resampy.resample(waveform, current_rate, sample_rate)
    return np.asarray(waveform, dtype=np.float32)
