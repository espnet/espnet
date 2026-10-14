"""Short-time objective intelligibility metric."""

from typing import Any

from espnet3.api.inference import Field
from espnet3.systems.esp2_enh.metrics.sisnr import MeanScore, waveform

try:
    from pystoi import stoi as stoi_fn
except ImportError:
    stoi_fn = None


class STOI(MeanScore):
    """Compute mean STOI or ESTOI. Requires: `pip install pystoi`.

    Args:
        fs: Sampling rate in Hz.
        extended: Whether to compute extended STOI (ESTOI). The result is
            then reported as ``ESTOI`` instead of ``STOI``.

    Examples:
        >>> metric = STOI(fs=16000)
        >>> metric.update(ref=["clean.wav"], hyp=["enhanced.wav"])
        >>> metric.compute()
        {'STOI': 0.937}
    """

    def __init__(self, fs: int = 16000, extended: bool = False):
        """Initialize STOI scoring with the configured options."""
        self.fs = fs
        self.extended = extended
        self.outputs = (Field("ESTOI" if extended else "STOI", "number"),)
        super().__init__()

    def score(self, reference: Any, estimate: Any) -> float:
        """Return the STOI of one utterance, cut to the shorter signal."""
        if stoi_fn is None:
            raise RuntimeError("STOI requires: `pip install pystoi`.")
        reference = waveform(reference, self.fs)
        estimate = waveform(estimate, self.fs)
        length = min(len(reference), len(estimate))
        return stoi_fn(
            reference[:length], estimate[:length], self.fs, extended=self.extended
        )
