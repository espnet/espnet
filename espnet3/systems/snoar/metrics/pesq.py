"""Wide-band PESQ (ITU-T P.862.2)."""

from __future__ import annotations

from espnet3.systems.snoar.metrics.audio_pair import AudioPairMetric


class PESQ(AudioPairMetric):
    """Mean wide-band PESQ of enhanced files (16 kHz) against references.

    Raises:
        ImportError: If the ``pesq`` package is not installed.
        ValueError: If the audio is not 16 kHz.
    """

    name = "pesq_wb"

    def __init__(self) -> None:
        """Check the dependency."""
        try:
            from pesq import pesq  # noqa: F401
        except ImportError as e:
            raise ImportError("PESQ needs `pip install pesq`") from e
        super().__init__()

    def score(self, ref, hyp, fs):
        """Return the PESQ of one utterance."""
        from pesq import pesq

        if fs != 16000:
            raise ValueError(f"wide-band PESQ needs 16 kHz audio, got {fs} Hz")
        return pesq(fs, ref, hyp, "wb")
