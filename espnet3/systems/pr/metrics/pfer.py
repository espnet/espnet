"""Phone feature error rate metric utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.systems.pr.metrics.scoring_utils import (
    clean_ipa_text,
    get_distance,
    segment_ipa,
)


class PFER(BaseMetric):
    """Compute the phone feature error rate over IPA transcripts.

    PFER is the articulatory-feature edit distance between the hypothesis and
    reference, divided by the number of reference phones. Unlike PER, a
    substitution costs the fraction of phonological features that differ rather
    than a flat 1, so confusing ``s`` with ``z`` is penalized far less than
    confusing it with an unrelated phone. PFER is therefore normally lower than
    PER on the same data, and it separates near-misses from real errors.

    The score is micro-averaged: distances and reference phone counts are pooled
    over the whole test set before dividing. The denominator is the same one PER
    uses, so the two are directly comparable.

    Select it from ``conf/metrics.yaml`` with::

        metrics:
          - metric:
              _target_: espnet3.systems.pr.metrics.pfer.PFER
    """

    def __init__(self, ref_key: str = "ref", hyp_key: str = "hyp") -> None:
        """Initialize the PFER metric.

        Args:
            ref_key: Key name for reference transcript entries, resolved to
                ``<inference_dir>/<test_name>/<ref_key>.scp``.
            hyp_key: Key name for hypothesis transcript entries.
        """
        self.ref_key = ref_key
        self.hyp_key = hyp_key

    def __call__(
        self,
        data: Dict[str, Path],
        test_name: str,
        inference_dir: Path,
    ) -> Dict[str, float]:
        """Compute PFER and return the score.

        Args:
            data: Mapping with ``data[self.ref_key]`` and ``data[self.hyp_key]``
                SCP files, aligned by utterance ID.
            test_name: Test set name. Unused, accepted for the metric contract.
            inference_dir: Base inference directory. Unused, accepted for the
                metric contract.

        Returns:
            ``{"PFER": <percentage>}``, or ``{"PFER": 0.0}`` when no reference
            phone is recognizable at all.

        Raises:
            RuntimeError: If ``panphon`` is not installed.
            AssertionError: If the two SCP files are not aligned by utterance ID.
        """
        distance = get_distance()
        total_distance = 0.0
        total_phones = 0
        for _, row in self.iter_inputs(data, self.ref_key, self.hyp_key):
            ref = clean_ipa_text(row[self.ref_key])
            hyp = clean_ipa_text(row[self.hyp_key])
            # feature_edit_distance segments internally, so it takes the
            # normalized strings while the denominator needs explicit phones.
            total_distance += distance.feature_edit_distance(hyp, ref)
            total_phones += len(segment_ipa(ref))

        if total_phones == 0:
            return {"PFER": 0.0}
        return {"PFER": round(total_distance / total_phones * 100, 1)}
