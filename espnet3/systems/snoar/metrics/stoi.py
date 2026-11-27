"""Short-time objective intelligibility."""

from __future__ import annotations

from espnet3.systems.snoar.metrics.audio_pair import AudioPairMetric


class STOI(AudioPairMetric):
    """Mean STOI of enhanced files against references.

    Raises:
        ImportError: If the ``pystoi`` package is not installed.
    """

    name = "stoi"

    def __init__(self) -> None:
        """Check the dependency."""
        try:
            import pystoi  # noqa: F401
        except ImportError as e:
            raise ImportError("STOI needs `pip install pystoi`") from e
        super().__init__()

    def score(self, ref, hyp, fs):
        """Return the STOI of one utterance."""
        from pystoi import stoi

        return stoi(ref, hyp, fs, extended=False)
