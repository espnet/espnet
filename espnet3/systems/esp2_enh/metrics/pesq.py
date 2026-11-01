"""Perceptual evaluation of speech quality metric."""

from typing import Any

from espnet3.contract import Field
from espnet3.systems.esp2_enh.metrics.sisnr import MeanScore, waveform

try:
    from pesq import pesq as pesq_fn
except ImportError:
    pesq_fn = None


class PESQ(MeanScore):
    """Compute mean PESQ. Requires: `pip install pesq`.

    Wide-band PESQ at 16 kHz, narrow-band at 8 kHz.

    Args:
        fs: Sampling rate. Must be 8000 or 16000 Hz.

    Examples:
        >>> metric = PESQ(fs=16000)
        >>> metric.update(ref=["clean.wav"], hyp=["enhanced.wav"])
        >>> metric.compute()
        {'PESQ': 2.551}
    """

    outputs = (Field("PESQ", "number"),)

    def __init__(self, fs: int = 16000):
        """Initialize PESQ scoring at the given sample rate."""
        if fs not in (8000, 16000):
            raise ValueError(f"PESQ supports only 8000 or 16000 Hz, but got {fs}")
        self.fs = fs
        super().__init__()

    def score(self, reference: Any, estimate: Any) -> float:
        """Return the PESQ of one utterance, cut to the shorter signal."""
        if pesq_fn is None:
            raise RuntimeError("PESQ requires: `pip install pesq`.")
        reference = waveform(reference, self.fs)
        estimate = waveform(estimate, self.fs)
        length = min(len(reference), len(estimate))
        mode = "wb" if self.fs == 16000 else "nb"
        return pesq_fn(self.fs, reference[:length], estimate[:length], mode)
