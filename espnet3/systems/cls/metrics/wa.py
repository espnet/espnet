"""Weighted accuracy metric for classification."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from espnet3.components.metrics.base_metric import BaseMetric


class WA(BaseMetric):
    """Compute weighted accuracy (WA) for a classification dataset.

    WA is the fraction of utterances whose predicted label equals the
    reference label. It is the same quantity espnet2's ``cls_score.py``
    reports as ``mean_acc``; the "mean" there averages one shared accuracy
    value across classes and is a no-op.
    """

    def __init__(self, ref_key: str = "ref", hyp_key: str = "hyp") -> None:
        """Initialize the WA metric.

        Args:
            ref_key: Key name for reference label entries.
            hyp_key: Key name for hypothesis label entries.
        """
        self.ref_key = ref_key
        self.hyp_key = hyp_key

    def __call__(
        self,
        data: Dict[str, Path],
        test_name: str,
        inference_dir: Path,
    ) -> Dict[str, float]:
        """Compute weighted accuracy.

        Args:
            data (Dict[str, Path]): Mapping of metric input aliases to SCP
                paths. Expects ``data[self.ref_key]`` and ``data[self.hyp_key]``
                to be aligned by utterance ID.
            test_name (str): Test set name. Unused by this metric.
            inference_dir (Path): Inference output root. Unused by this metric.

        Returns:
            Dict[str, float]: ``{"WA": <percentage>}``

        Raises:
            ValueError: If the reference SCP is empty.
            AssertionError: If the SCP files are not aligned by utterance ID.

        Example:
            >>> metric({"ref": Path("test/ref.scp"), "hyp": Path("test/hyp.scp")},
            ...        "test", Path("infer"))
            {'WA': 50.77}
        """
        total = 0
        correct = 0
        for _, row in self.iter_inputs(data, self.ref_key, self.hyp_key):
            total += 1
            correct += int(row[self.ref_key] == row[self.hyp_key])
        if total == 0:
            raise ValueError(f"No utterances to score in {data[self.ref_key]}")
        return {"WA": round(correct / total * 100, 2)}
