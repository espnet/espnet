"""SLU metrics for the SLURP recipe."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from espnet3.components.metrics.base_metric import BaseMetric


class IntentAccuracy(BaseMetric):
    """Share of utterances whose predicted intent label matches the reference.

    This is the headline SLURP number, the one ``egs2/slurp/asr1`` reports: the
    predicted intent is the first token of the inference hypothesis, and an
    utterance counts as correct only on an exact label match. ``src/inference.py``
    has already split that token off, so this metric reads the dedicated
    ``*_intent`` SCP fields rather than re-parsing the full hypothesis.

    Stages: instantiated by ``measure`` from ``metrics.yaml``.

    Args:
        ref_key: SCP field holding the reference intent label.
        hyp_key: SCP field holding the predicted intent label.
        write_errors: Whether to write the misclassified utterances to
            ``<inference_dir>/<test_name>/intent_errors`` for error analysis.

    Examples:
        In ``metrics.yaml``::

            metrics:
              - metric:
                  _target_: src.metrics.IntentAccuracy
                  ref_key: ref_intent
                  hyp_key: hyp_intent
    """

    def __init__(
        self,
        ref_key: str = "ref_intent",
        hyp_key: str = "hyp_intent",
        write_errors: bool = True,
    ) -> None:
        self.ref_key = ref_key
        self.hyp_key = hyp_key
        self.write_errors = write_errors

    def __call__(
        self,
        data: Dict[str, Path],
        test_name: str,
        inference_dir: Path,
    ) -> Dict[str, float]:
        """Score one test set and optionally dump its misclassified utterances.

        Args:
            data: Metric input aliases mapped to the SCP files ``measure``
                resolved, which must include ``ref_key`` and ``hyp_key``.
            test_name: Name of the test set, used as the output subdirectory.
            inference_dir: Directory holding the per-test-set inference output.

        Returns:
            ``{"IntentAccuracy": <percentage rounded to two decimals>}``.

        Raises:
            RuntimeError: If the test set holds no utterance.
            AssertionError: If the two SCP files are not aligned by ID, which
                ``BaseMetric.iter_inputs`` checks row by row.
        """
        total = 0
        correct = 0
        errors: list[str] = []

        for utt_id, row in self.iter_inputs(data, self.ref_key, self.hyp_key):
            reference = row[self.ref_key].strip()
            hypothesis = row[self.hyp_key].strip()
            total += 1
            if reference == hypothesis:
                correct += 1
            else:
                errors.append(f"{utt_id}\t{reference}\t{hypothesis}")

        if total == 0:
            raise RuntimeError(f"No utterance found for test set '{test_name}'.")

        if self.write_errors:
            test_dir = Path(inference_dir) / test_name
            test_dir.mkdir(parents=True, exist_ok=True)
            lines = ["utt_id\tref_intent\thyp_intent", *errors]
            (test_dir / "intent_errors").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )

        return {"IntentAccuracy": round(100 * correct / total, 2)}
