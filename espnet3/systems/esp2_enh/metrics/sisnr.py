"""Scale-invariant signal-to-noise ratio metric."""

from pathlib import Path
from typing import Dict

import librosa
import numpy as np
import soundfile as sf
import torch

from espnet2.enh.loss.criterions.time_domain import SISNRLoss
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
    """Compute SI-SNR between one reference and estimated waveform.

    Uses the same ``SISNRLoss`` as ``espnet2/bin/enh_scoring.py`` so scores match
    ESPnet2. Both signals are cut to the shorter length first.
    """
    length = min(len(reference), len(estimate))
    reference = torch.from_numpy(np.ascontiguousarray(reference[:length]))
    estimate = torch.from_numpy(np.ascontiguousarray(estimate[:length]))
    return -float(SISNRLoss()(reference[None], estimate[None]))


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
