"""Scale-invariant signal-to-noise ratio metric."""

from pathlib import Path
from typing import Any, Dict, Sequence

import librosa
import numpy as np
import soundfile as sf
import torch

from espnet2.enh.loss.criterions.time_domain import SISNRLoss
from espnet3.api.inference import Field
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


def waveform(value: Any, sample_rate: int | None = None) -> np.ndarray:
    """Return one ``update`` value as a mono waveform.

    ``measure`` passes the paths an SCP lists; a training callback may pass
    the samples themselves, which are used as they are.
    """
    if isinstance(value, (str, Path)):
        return load_audio(value, sample_rate)
    return np.asarray(value, dtype=np.float32)


class MeanScore(BaseMetric):
    """Average a per-utterance score of ``ref`` and ``hyp`` over a test set.

    ``ref`` is the clean speech and ``hyp`` the enhanced speech. A subclass
    implements :meth:`score` and declares the one output it returns.
    """

    inputs = (Field("ref", "audio"), Field("hyp", "audio"))

    def __init__(self) -> None:
        """Register the running total and count of scored utterances."""
        super().__init__()
        self.add_state("total", 0.0, "sum")
        self.add_state("count", 0, "sum")

    def score(self, reference: np.ndarray, estimate: np.ndarray) -> float:
        """Return the score of one utterance."""
        raise NotImplementedError

    def update(self, *, ref: Sequence[Any], hyp: Sequence[Any]) -> None:
        """Add the scores of one batch of utterances."""
        for reference, estimate in zip(ref, hyp):
            self.total += float(self.score(reference, estimate))
            self.count += 1

    def compute(self) -> Dict[str, float]:
        """Return the mean score, rounded to 4 places.

        Raises:
            ValueError: If no utterance was scored.
        """
        if self.count == 0:
            raise ValueError("nothing to score: the test set is empty")
        return {self.outputs[0].name: round(self.total / self.count, 4)}


class SISNR(MeanScore):
    """Compute mean SI-SNR (dB) of enhanced speech against clean speech.

    Examples:
        >>> metric = SISNR()
        >>> metric.update(ref=["clean.wav"], hyp=["enhanced.wav"])
        >>> metric.compute()
        {'SI_SNR': 18.73}
    """

    outputs = (Field("SI_SNR", "number"),)

    def score(self, reference: Any, estimate: Any) -> float:
        """Return the SI-SNR of one utterance."""
        return si_snr(waveform(reference), waveform(estimate))
