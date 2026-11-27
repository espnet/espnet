"""Shared base for metrics that compare an enhanced file with a reference."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import soundfile as sf

from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.contract import Field


def read_pair(ref_path: str, hyp_path: str):
    """Read a reference and an estimate, trimmed to the shorter length.

    Raises:
        ValueError: If the two files have different sampling rates.
    """
    ref, fs_r = sf.read(ref_path, dtype="float32", always_2d=True)
    hyp, fs_h = sf.read(hyp_path, dtype="float32", always_2d=True)
    if fs_r != fs_h:
        raise ValueError(f"{ref_path} is {fs_r} Hz but {hyp_path} is {fs_h} Hz")
    n = min(len(ref), len(hyp))
    return ref[:n, 0].astype(np.float64), hyp[:n, 0].astype(np.float64), fs_r


class AudioPairMetric(BaseMetric):
    """Average a per-utterance score of ``(ref, hyp)`` audio file paths.

    Subclasses set ``name`` and implement ``score(ref, hyp, fs)``. Bind the
    inputs in ``metrics.yaml``:

    .. code-block:: yaml

        inputs:
          ref: dataset:speech_ref
          hyp: enhanced
    """

    name = "score"
    inputs = (Field("ref", "text"), Field("hyp", "text"))

    def __init__(self) -> None:
        """Initialize the running sums."""
        self.outputs = (Field(self.name, "number"),)
        super().__init__()
        self.add_state("total", 0.0, "sum")
        self.add_state("count", 0, "sum")

    def score(self, ref: np.ndarray, hyp: np.ndarray, fs: int) -> float:
        """Return the score of one utterance."""
        raise NotImplementedError

    def update(self, *, ref: Sequence[str], hyp: Sequence[str]) -> None:
        """Score each pair of files and add it to the running sums."""
        for r, h in zip(ref, hyp):
            rr, hh, fs = read_pair(r, h)
            self.total += float(self.score(rr, hh, fs))
            self.count += 1

    def compute(self) -> Dict[str, float]:
        """Return the mean score, rounded to 3 places.

        Raises:
            ValueError: If no utterance was added.
        """
        if self.count == 0:
            raise ValueError("No utterances to score")
        return {self.name: round(self.total / self.count, 3)}
