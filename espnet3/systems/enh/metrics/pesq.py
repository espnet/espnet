"""Perceptual evaluation of speech quality metric."""

from pathlib import Path
from typing import Dict

import numpy as np

from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.systems.enh.metrics.audio import load_audio

try:
    from pesq import pesq as pesq_fn
except ImportError:
    pesq_fn = None


class PESQMetric(BaseMetric):
    """Compute PESQ. Requires: `pip install pesq`.

    Args:
        ref_key: Input alias for reference waveform paths.
        hyp_key: Input alias for enhanced waveform paths.
        fs: Sampling rate. Must be 8000 or 16000 Hz.
    """

    def __init__(
        self,
        ref_key: str = "reference",
        hyp_key: str = "enhanced",
        fs: int = 16000,
    ):
        """Initialize PESQ scoring for the configured SCP keys and sample rate."""
        if fs not in (8000, 16000):
            raise ValueError(f"PESQ supports only 8000 or 16000 Hz, but got {fs}")
        self.ref_key = ref_key
        self.hyp_key = hyp_key
        self.fs = fs

    def __call__(
        self, data: Dict[str, Path], test_name: str, inference_dir: Path
    ) -> Dict[str, float]:
        """Return the mean PESQ score for one test set."""
        if pesq_fn is None:
            raise RuntimeError("PESQ requires: `pip install pesq`.")
        mode = "wb" if self.fs == 16000 else "nb"
        scores = []
        for _, row in self.iter_inputs(data, self.ref_key, self.hyp_key):
            reference = load_audio(row[self.ref_key], self.fs)
            estimate = load_audio(row[self.hyp_key], self.fs)
            length = min(len(reference), len(estimate))
            scores.append(pesq_fn(self.fs, reference[:length], estimate[:length], mode))
        mean = float(np.mean(scores)) if scores else float("nan")
        return {"PESQ": round(mean, 4)}
