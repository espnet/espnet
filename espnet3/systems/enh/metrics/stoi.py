"""Short-time objective intelligibility metric."""

from pathlib import Path
from typing import Dict

import numpy as np

from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.systems.enh.metrics.audio import load_audio

try:
    from pystoi import stoi as stoi_fn
except ImportError:
    stoi_fn = None


class STOIMetric(BaseMetric):
    """Compute STOI. Requires: `pip install pystoi`.

    Args:
        ref_key: Input alias for reference waveform paths.
        hyp_key: Input alias for enhanced waveform paths.
        fs: Sampling rate in Hz.
        extended: Whether to compute extended STOI (ESTOI).
    """

    def __init__(
        self,
        ref_key: str = "reference",
        hyp_key: str = "enhanced",
        fs: int = 16000,
        extended: bool = False,
    ):
        self.ref_key = ref_key
        self.hyp_key = hyp_key
        self.fs = fs
        self.extended = extended

    def __call__(
        self, data: Dict[str, Path], test_name: str, inference_dir: Path
    ) -> Dict[str, float]:
        """Return the mean STOI or ESTOI score for one test set."""
        if stoi_fn is None:
            raise RuntimeError("STOI requires: `pip install pystoi`.")
        scores = []
        for _, row in self.iter_inputs(data, self.ref_key, self.hyp_key):
            reference = load_audio(row[self.ref_key], self.fs)
            estimate = load_audio(row[self.hyp_key], self.fs)
            length = min(len(reference), len(estimate))
            scores.append(
                stoi_fn(
                    reference[:length],
                    estimate[:length],
                    self.fs,
                    extended=self.extended,
                )
            )
        mean = float(np.mean(scores)) if scores else float("nan")
        label = "ESTOI" if self.extended else "STOI"
        return {label: round(mean, 4)}
