"""Macro precision, recall, F1 and per-language LID reports."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Sequence

from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.contract import Field


class ClassificationReport(BaseMetric):
    """Compute macro precision/recall/F1 and write per-language reports.

    Macro averages run over the languages in the references, as in ESPnet2
    LID scoring; a predicted language that no reference has is counted only
    as a false positive. With one language per utterance, micro precision,
    recall and F1 all equal :class:`Accuracy`'s ``MicroAccuracy``, so they
    are not reported again here.

    Examples:
        >>> metric = ClassificationReport()
        >>> metric.update(ref=["eng", "eng", "fra"], hyp=["eng", "fra", "fra"])
        >>> metric.compute()
        {'MacroPrecision': 75.0, 'MacroRecall': 75.0, 'MacroF1': 66.67}
    """

    inputs = (Field("ref", "text"), Field("hyp", "text"))
    outputs = (
        Field("MacroPrecision", "number"),
        Field("MacroRecall", "number"),
        Field("MacroF1", "number"),
    )

    def __init__(self) -> None:
        """Register the counts of reference/prediction pairs."""
        super().__init__()
        # One [reference, hypothesis, count] row per pair of each batch.
        self.add_state("pairs", [], "cat")

    @staticmethod
    def _percentage(value: float) -> float:
        """Round a fraction to a percentage for reports."""
        return round(value * 100, 2)

    def update(self, *, ref: Sequence[str], hyp: Sequence[str]) -> None:
        """Add the reference/prediction pair counts of one batch.

        Args:
            ref: Reference language codes, one per utterance.
            hyp: Predicted language codes, aligned with ``ref``.

        Raises:
            ValueError: If a label is empty.
        """
        pairs = Counter()
        for reference, hypothesis in zip(ref, hyp):
            reference, hypothesis = reference.strip(), hypothesis.strip()
            if not reference or not hypothesis:
                raise ValueError("Empty LID label")
            pairs[reference, hypothesis] += 1
        self.pairs.extend([r, h, n] for (r, h), n in pairs.items())

    def _counts(self) -> Counter:
        """Return the summed count of each (reference, hypothesis) pair."""
        pairs = Counter()
        for reference, hypothesis, n in self.pairs:
            pairs[reference, hypothesis] += n
        return pairs

    def _per_language(self) -> Dict[str, Dict[str, float]]:
        """Return the counts and percentages of each reference language.

        Raises:
            ValueError: If no prediction was added.
        """
        target_total, true_positive, false_positive = Counter(), Counter(), Counter()
        for (reference, hypothesis), n in self._counts().items():
            target_total[reference] += n
            if reference == hypothesis:
                true_positive[reference] += n
            else:
                false_positive[hypothesis] += n
        if not target_total:
            raise ValueError("No LID predictions found")

        per_language = {}
        for language in sorted(target_total):
            hits = true_positive[language]
            predicted = hits + false_positive[language]
            precision = hits / predicted if predicted else 0.0
            recall = hits / target_total[language]
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            per_language[language] = {
                "Count": target_total[language],
                "Correct": hits,
                "Accuracy": recall,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
            }
        return per_language

    def compute(self) -> Dict[str, float]:
        """Return macro-averaged percentages, rounded to two decimals.

        Raises:
            ValueError: If no prediction was added.
        """
        per_language = self._per_language().values()
        return {
            f"Macro{name}": self._percentage(
                sum(scores[name] for scores in per_language) / len(per_language)
            )
            for name in ("Precision", "Recall", "F1")
        }

    def write_artifacts(self, test_dir: Path) -> None:
        """Write ``lid_per_language.json`` and ``lid_error_counts.json``.

        The first holds each reference language's counts and percentages, the
        second each confusion such as ``"deu->eng"`` with its count, most
        frequent first. Re-running replaces both.

        Args:
            test_dir: The test set's output directory, already created.
        """
        per_language = {
            language: {
                name: value if name in ("Count", "Correct") else self._percentage(value)
                for name, value in scores.items()
            }
            for language, scores in self._per_language().items()
        }
        error_counts = {
            f"{reference}->{hypothesis}": n
            for (reference, hypothesis), n in sorted(
                self._counts().items(), key=lambda item: (-item[1], item[0])
            )
            if reference != hypothesis
        }
        for name, report in (
            ("lid_per_language.json", per_language),
            ("lid_error_counts.json", error_counts),
        ):
            (test_dir / name).write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
