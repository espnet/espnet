"""Weighted accuracy metric for classification."""

from __future__ import annotations

from typing import Dict, Sequence

from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.contract import Field


class WA(BaseMetric):
    """Compute weighted accuracy (WA) for a classification dataset.

    WA is the fraction of utterances whose predicted label equals the
    reference label. It is the same quantity espnet2's ``cls_score.py``
    reports as ``mean_acc``; the "mean" there averages one shared accuracy
    value across classes and is a no-op.

    Declare it in ``conf/metrics.yaml``:

    .. code-block:: yaml

        - metric:
            _target_: espnet3.systems.esp2_cls.metrics.wa.WA
          inputs:
            ref: dataset:label
            hyp: hyp
    """

    inputs = (Field("ref", "text"), Field("hyp", "text"))
    outputs = (Field("WA", "number"),)

    def __init__(self) -> None:
        """Initialize the WA metric."""
        super().__init__()
        self.add_state("total", 0, "sum")
        self.add_state("correct", 0, "sum")

    def update(self, *, ref: Sequence[str], hyp: Sequence[str]) -> None:
        """Add the correct and total counts of one batch to the state.

        Args:
            ref: Reference labels, one per utterance.
            hyp: Predicted labels, aligned with ``ref``.

        Examples:
            >>> metric = WA()
            >>> metric.update(ref=["joy", "anger"], hyp=["joy", "joy"])
        """
        self.total += len(ref)
        self.correct += sum(int(r == h) for r, h in zip(ref, hyp))

    def compute(self) -> Dict[str, float]:
        """Return the weighted accuracy of everything added so far.

        Returns:
            Dict[str, float]: ``{"WA": <percentage>}``, rounded to 2 places.

        Raises:
            ValueError: If no utterance was added.

        Examples:
            >>> metric = WA()
            >>> metric.update(ref=["joy", "anger"], hyp=["joy", "joy"])
            >>> metric.compute()
            {'WA': 50.0}
        """
        if self.total == 0:
            raise ValueError("No utterances to score")
        return {"WA": round(self.correct / self.total * 100, 2)}
