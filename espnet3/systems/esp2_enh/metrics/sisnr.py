"""Scale-invariant signal-to-noise ratio metric."""

from pathlib import Path
from typing import Dict

import librosa
import numpy as np
import soundfile as sf

from espnet3.components.metrics.base_metric import BaseMetric


def load_audio(path: str | Path, sample_rate: int | None = None) -> np.ndarray:
    """Load mono audio and optionally resample it.

    Also used by the PESQ and STOI metrics in this package.

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
        waveform = librosa.resample(
            waveform, orig_sr=current_rate, target_sr=sample_rate
        )
    return np.asarray(waveform, dtype=np.float32)


def si_snr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Compute SI-SNR between one reference and estimated waveform."""
    length = min(len(reference), len(estimate))
    reference = reference[:length] - reference[:length].mean()
    estimate = estimate[:length] - estimate[:length].mean()
    target = (
        np.dot(estimate, reference) / (np.dot(reference, reference) + 1e-8)
    ) * reference
    noise = estimate - target
    return float(
        10.0 * np.log10((np.dot(target, target) + 1e-8) / (np.dot(noise, noise) + 1e-8))
    )


class SISNR(BaseMetric):
    """Compute mean SI-SNR from aligned reference and enhanced WAV SCPs.

    Args:
        ref_key: Input alias for reference waveform paths.
        hyp_key: Input alias for enhanced waveform paths.
    """

    def __init__(self, ref_key: str = "reference", hyp_key: str = "enhanced"):
        """Initialize SI-SNR scoring for the configured SCP keys."""
        self.ref_key = ref_key
        self.hyp_key = hyp_key

    def __call__(
        self, data: Dict[str, Path], test_name: str, inference_dir: Path
    ) -> Dict[str, float]:
        """Return the mean SI-SNR for one test set."""
        scores = []
        for _, row in self.iter_inputs(data, self.ref_key, self.hyp_key):
            scores.append(
                si_snr(load_audio(row[self.ref_key]), load_audio(row[self.hyp_key]))
            )
        mean = float(np.mean(scores)) if scores else float("nan")
        return {"SI-SNR": round(mean, 4)}
