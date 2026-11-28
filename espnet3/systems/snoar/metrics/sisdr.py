"""Scale-invariant SDR."""

from __future__ import annotations

import numpy as np

from espnet3.systems.snoar.metrics.audio_pair import AudioPairMetric


def compute_si_sdr_np(ref: np.ndarray, hyp: np.ndarray, eps: float = 1e-8) -> float:
    """Return SI-SDR in dB of ``hyp`` against ``ref`` (no mean removal).

    Examples:
        >>> x = np.random.default_rng(0).standard_normal(1000)
        >>> compute_si_sdr_np(x, 2 * x) > 60
        True
    """
    alpha = np.dot(hyp, ref) / (np.dot(ref, ref) + eps)
    proj = alpha * ref
    err = hyp - proj
    return float(10 * np.log10(np.dot(proj, proj) / (np.dot(err, err) + eps) + eps))


class SISDR(AudioPairMetric):
    """Mean SI-SDR (dB) of enhanced files against their references."""

    name = "si_sdr"

    def score(self, ref, hyp, fs):
        """Return the SI-SDR of one utterance."""
        return compute_si_sdr_np(ref, hyp)
