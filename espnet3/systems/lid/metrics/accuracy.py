"""Language identification accuracy metrics."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict

from espnet3.components.metrics.base_metric import BaseMetric


class Accuracy(BaseMetric):
    """Compute overall and macro-averaged LID classification metrics."""

    def __init__(self, ref_key: str = "ref", hyp_key: str = "hyp") -> None:
        """Select the aligned reference and hypothesis SCP aliases."""
        self.ref_key = ref_key
        self.hyp_key = hyp_key

    @staticmethod
    def _percentage(value: float) -> float:
        return round(value * 100, 2)

    def __call__(
        self,
        data: Dict[str, Path],
        test_name: str,
        inference_dir: Path,
    ) -> Dict[str, float]:
        """Score aligned reference and predicted ISO 639-3 language codes.

        Args:
            data: Reference and hypothesis SCP paths selected by the metric keys.
            test_name: Name of the inference test set.
            inference_dir: Root inference directory. Per-test-set outputs are
                ``lid_errors``, ``lid_per_language.json`` and
                ``lid_error_counts.json``. Re-running replaces these summaries.

        Returns:
            Overall and macro-averaged percentages, using the ESPnet2 LID
            convention of averaging across languages in the reference set.

        Raises:
            ValueError: If labels are empty or there are no predictions.
            AssertionError: If reference and hypothesis SCP IDs are not aligned.
        """
        rows = []
        for utt_id, row in self.iter_inputs(data, self.ref_key, self.hyp_key):
            reference = row[self.ref_key].strip()
            hypothesis = row[self.hyp_key].strip()
            if not reference or not hypothesis:
                raise ValueError(f"Empty LID label found for utterance: {utt_id}")
            rows.append((utt_id, reference, hypothesis))
        if not rows:
            raise ValueError("No LID predictions found")

        target_total = Counter(reference for _, reference, _ in rows)
        true_positive = Counter()
        false_positive = Counter()
        for _, reference, hypothesis in rows:
            if reference == hypothesis:
                true_positive[reference] += 1
            else:
                false_positive[hypothesis] += 1

        target_languages = sorted(target_total)
        correct = sum(true_positive.values())
        total = len(rows)

        recalls = []
        precisions = []
        f1_scores = []
        per_language = {}
        for language in target_languages:
            true_positive_count = true_positive[language]
            precision = (
                true_positive_count / (true_positive_count + false_positive[language])
                if true_positive_count + false_positive[language]
                else 0.0
            )
            recall = true_positive_count / target_total[language]
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            per_language[language] = {
                "Count": target_total[language],
                "Correct": true_positive_count,
                "Accuracy": self._percentage(recall),
                "Precision": self._percentage(precision),
                "Recall": self._percentage(recall),
                "F1": self._percentage(f1),
            }

        test_dir = Path(inference_dir) / test_name
        test_dir.mkdir(parents=True, exist_ok=True)
        errors = [
            f"{utt_id} {reference} {hypothesis}\n"
            for utt_id, reference, hypothesis in rows
            if reference != hypothesis
        ]
        (test_dir / "lid_errors").write_text("".join(errors), encoding="utf-8")
        (test_dir / "lid_per_language.json").write_text(
            json.dumps(per_language, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        error_counts = Counter(
            f"{reference}->{hypothesis}"
            for _, reference, hypothesis in rows
            if reference != hypothesis
        )
        (test_dir / "lid_error_counts.json").write_text(
            json.dumps(
                dict(
                    sorted(error_counts.items(), key=lambda item: (-item[1], item[0]))
                ),
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        accuracy = correct / total
        return {
            "Accuracy": self._percentage(accuracy),
            "Precision": self._percentage(accuracy),
            "Recall": self._percentage(accuracy),
            "F1": self._percentage(accuracy),
            "Macro Accuracy": self._percentage(sum(recalls) / len(recalls)),
            "Macro Precision": self._percentage(sum(precisions) / len(precisions)),
            "Macro Recall": self._percentage(sum(recalls) / len(recalls)),
            "Macro F1": self._percentage(sum(f1_scores) / len(f1_scores)),
        }
