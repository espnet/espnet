"""The shared base of the metrics that score verification trials."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

from espnet3.api import Field
from espnet3.components.metrics.base_metric import BaseMetric


class TrialMetric(BaseMetric):
    """Collect every trial's label and score, for a metric over all of them.

    EER and minDCF are not sums over trials: each needs the whole score
    distribution at once, so ``update`` only keeps the trials and the
    subclass's ``compute`` reads :attr:`labels` and :attr:`scores`.

    Inputs, bound in ``metrics.yaml``::

        inputs:
          label: dataset:spk_labels   # 1 for a target trial, 0 for nontarget
          score: score                # the ``Inference`` output

    Both are numbers, but ``measure`` hands a metric its SCP values as
    strings, so :meth:`__call__` parses them before :meth:`update`, which
    checks them as the ``number`` kind like every other caller's.
    """

    inputs = (Field("label", "number"), Field("score", "number"))

    def __init__(self) -> None:
        """Register the trial states and check the declaration."""
        super().__init__()
        self.add_state("labels", [], "cat")
        self.add_state("scores", [], "cat")

    def update(self, *, label: Sequence[int], score: Sequence[float]) -> None:
        """Keep a chunk of trials."""
        self.labels.extend(int(x) for x in label)
        self.scores.extend(float(x) for x in score)

    def __call__(
        self, data: Dict[str, Path], test_name: str, output_dir: Optional[Path] = None
    ) -> Dict[str, float]:
        """Score one test set, parsing the SCP strings into numbers first.

        Args:
            data: ``{"label": <scp>, "score": <scp>}``, aligned by trial.
            test_name: Test set name, the artifact subdirectory.
            output_dir: Root directory for artifacts; ``None`` writes none.

        Returns:
            What :meth:`compute` returns.
        """
        self.reset()
        self.collect_artifacts = output_dir is not None
        try:
            labels: List[int] = []
            scores: List[float] = []
            for _, row in self.iter_inputs(data, "label", "score"):
                labels.append(int(row["label"]))
                scores.append(float(row["score"]))
            self.update(label=labels, score=scores)
            result = self.compute()
            if output_dir is not None:
                test_dir = Path(output_dir) / test_name
                test_dir.mkdir(parents=True, exist_ok=True)
                self.write_artifacts(test_dir)
            return result
        finally:
            self.collect_artifacts = False
