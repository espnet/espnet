"""Minimum detection cost metric for speaker verification trials."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.systems.spk.scoring import compute_min_dcf


class MinDCF(BaseMetric):
    """Compute the normalized minimum detection cost over scored trials.

    Args:
        ref_key: Key of the trial label entries.
        hyp_key: Key of the trial score entries.
        p_target: Prior probability of a target trial.
        c_miss: Cost of a missed detection.
        c_fa: Cost of a false alarm.

    Notes:
        VoxCeleb results are conventionally reported at ``p_target=0.05`` with
        unit costs, which is what the defaults use.
    """

    def __init__(
        self,
        ref_key: str = "label",
        hyp_key: str = "score",
        p_target: float = 0.05,
        c_miss: float = 1.0,
        c_fa: float = 1.0,
    ) -> None:
        """Initialize the minDCF metric with its operating point."""
        self.ref_key = ref_key
        self.hyp_key = hyp_key
        self.p_target = float(p_target)
        self.c_miss = float(c_miss)
        self.c_fa = float(c_fa)

    def __call__(
        self,
        data: Dict[str, Path],
        test_name: str,
        inference_dir: Path,
    ) -> Dict[str, float]:
        """Compute minDCF for one test set.

        Args:
            data: Mapping of metric input aliases to SCP paths. Expects
                ``data[self.ref_key]`` and ``data[self.hyp_key]`` to be aligned
                by trial ID.
            test_name: Test set name, unused but part of the metric interface.
            inference_dir: Base inference directory, unused by this metric.

        Returns:
            ``{"minDCF": <cost>}``.

        Raises:
            AssertionError: If the score and label SCP files are not aligned.
        """
        scores = []
        labels = []
        for _, row in self.iter_inputs(data, self.ref_key, self.hyp_key):
            scores.append(float(row[self.hyp_key]))
            labels.append(int(row[self.ref_key]))

        min_dcf = compute_min_dcf(
            scores,
            labels,
            p_target=self.p_target,
            c_miss=self.c_miss,
            c_fa=self.c_fa,
        )
        return {"minDCF": round(min_dcf, 5)}
