"""Intent classification accuracy for the esp2_slu system."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.contract import Field


class IntentAccuracy(BaseMetric):
    """Share of utterances whose predicted intent label matches the reference.

    This is the headline SLURP number, the one ``egs2/slurp/slu1`` reports: an
    utterance counts as correct only on an exact label match. The system's
    inference runner has already split the label off the decoded sequence, so
    this metric compares two plain strings and never re-parses a hypothesis.

    Stages: instantiated by ``measure`` from ``metrics.yaml``.

    Args:
        write_errors: Whether to write the misclassified utterances to
            ``<test_dir>/intent_errors`` for error analysis.

    Examples:
        In ``metrics.yaml``::

            metrics:
              - metric:
                  _target_: espnet3.systems.esp2_slu.metrics.intent_accuracy.IntentAccuracy
                inputs:
                  ref: dataset:intent
                  hyp: hyp_intent
    """  # noqa: E501

    inputs = (Field("ref", "text"), Field("hyp", "text"))
    outputs = (Field("IntentAccuracy", "number"),)

    def __init__(self, write_errors: bool = True) -> None:
        """Set up the pooled counts and the optional error listing."""
        super().__init__()
        self.write_errors = write_errors
        self.add_state("correct", 0, "sum")
        self.add_state("total", 0, "sum")
        self._errors: List[str] = []

    def reset(self) -> None:
        """Reset the counts and drop errors kept from an unfinished run."""
        super().reset()
        self._errors = []

    def update(self, *, ref: Sequence[str], hyp: Sequence[str]) -> None:
        """Count matches over one batch of label pairs.

        ``update`` is not given the sample IDs, and this recipe's IDs are the
        dataset indices, so the rows are numbered with the running total. That
        keeps ``intent_errors`` aligned with the SCP files as long as the
        batches arrive in dataset order, which is how ``measure`` reads them.

        Args:
            ref: Reference intent labels.
            hyp: Predicted intent labels.
        """
        for offset, (reference, hypothesis) in enumerate(zip(ref, hyp)):
            reference = reference.strip()
            hypothesis = hypothesis.strip()
            if reference == hypothesis:
                self.correct += 1
            elif self.collect_artifacts and self.write_errors:
                self._errors.append(f"{self.total + offset}\t{reference}\t{hypothesis}")
        self.total += len(ref)

    def compute(self) -> Dict[str, float]:
        """Return the accuracy over every utterance seen since the last reset.

        Raises:
            RuntimeError: If no utterance was scored.
        """
        if self.total == 0:
            raise RuntimeError("No utterance to score.")
        return {"IntentAccuracy": round(100.0 * self.correct / self.total, 2)}

    def write_artifacts(self, test_dir: Path) -> None:
        """Write the misclassified utterances to ``test_dir/intent_errors``.

        Args:
            test_dir: The test set's output directory, already created.
        """
        if not self.write_errors:
            return
        lines = ["idx\tref_intent\thyp_intent", *self._errors]
        (test_dir / "intent_errors").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        self._errors = []
