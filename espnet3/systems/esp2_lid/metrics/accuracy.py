"""Micro and macro language identification accuracy."""

from collections import Counter
from typing import Dict, Sequence

from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.contract import Field


class Accuracy(BaseMetric):
    """Compute accuracy over utterances and average recall over languages.

    ``MicroAccuracy`` counts every utterance once; ``MacroAccuracy`` averages
    the accuracy of each language in the references, so small languages
    weigh as much as large ones.

    Examples:
        >>> metric = Accuracy()
        >>> metric.update(ref=["eng", "eng", "fra"], hyp=["eng", "fra", "fra"])
        >>> metric.compute()
        {'MicroAccuracy': 66.67, 'MacroAccuracy': 75.0}
    """

    inputs = (Field("ref", "text"), Field("hyp", "text"))
    outputs = (Field("MicroAccuracy", "number"), Field("MacroAccuracy", "number"))

    def __init__(self) -> None:
        """Register the per-language counts."""
        super().__init__()
        # One [language, count, correct] row per language of each batch.
        self.add_state("languages", [], "cat")

    def update(self, *, ref: Sequence[str], hyp: Sequence[str]) -> None:
        """Add the utterance and correct counts of one batch, per language.

        Args:
            ref: Reference language codes, one per utterance.
            hyp: Predicted language codes, aligned with ``ref``.

        Raises:
            ValueError: If a label is empty.
        """
        count, correct = Counter(), Counter()
        for reference, hypothesis in zip(ref, hyp):
            reference, hypothesis = reference.strip(), hypothesis.strip()
            if not reference or not hypothesis:
                raise ValueError("Empty LID label")
            count[reference] += 1
            correct[reference] += reference == hypothesis
        self.languages.extend(
            [language, n, correct[language]] for language, n in count.items()
        )

    def compute(self) -> Dict[str, float]:
        """Return micro and macro accuracy percentages, rounded to two decimals.

        Raises:
            ValueError: If no prediction was added.
        """
        count, correct = Counter(), Counter()
        for language, n, n_correct in self.languages:
            count[language] += n
            correct[language] += n_correct
        if not count:
            raise ValueError("No LID predictions found")
        return {
            "MicroAccuracy": round(
                100 * sum(correct.values()) / sum(count.values()), 2
            ),
            "MacroAccuracy": round(
                100 * sum(correct[k] / n for k, n in count.items()) / len(count), 2
            ),
        }
