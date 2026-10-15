"""Minimum detection cost metric for speaker verification trials."""

from __future__ import annotations

from typing import Dict

from espnet3.api import Field
from espnet3.systems.esp2_spk.metrics.trials import TrialMetric
from espnet3.systems.esp2_spk.src.scoring import compute_min_dcf


class MinDCF(TrialMetric):
    """Compute the normalized minimum detection cost over scored trials.

    Args:
        p_target: Prior probability of a target trial.
        c_miss: Cost of a missed detection.
        c_fa: Cost of a false alarm.

    Notes:
        VoxCeleb results are conventionally reported at ``p_target=0.05`` with
        unit costs, which is what the defaults use.
    """

    outputs = (Field("minDCF", "number"),)

    def __init__(
        self,
        p_target: float = 0.05,
        c_miss: float = 1.0,
        c_fa: float = 1.0,
    ) -> None:
        """Initialize the minDCF metric with its operating point."""
        self.p_target = float(p_target)
        self.c_miss = float(c_miss)
        self.c_fa = float(c_fa)
        super().__init__()

    def compute(self) -> Dict[str, float]:
        """Return ``{"minDCF": <cost>}``."""
        min_dcf = compute_min_dcf(
            self.scores,
            self.labels,
            p_target=self.p_target,
            c_miss=self.c_miss,
            c_fa=self.c_fa,
        )
        return {"minDCF": round(min_dcf, 5)}
