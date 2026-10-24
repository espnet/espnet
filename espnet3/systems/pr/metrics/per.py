"""Phone error rate metric utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

try:
    import jiwer
except ImportError:
    jiwer = None

from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.systems.pr.metrics.scoring_utils import clean_ipa_text, segment_ipa


class PER(BaseMetric):
    """Compute the phone error rate over IPA transcripts.

    PER is the edit distance between the hypothesis and reference phone
    sequences, divided by the number of reference phones. The score is
    micro-averaged: errors and reference phones are pooled over the whole test
    set before dividing, so long utterances weigh more than short ones.

    Both sides are normalized with
    :func:`espnet3.systems.pr.metrics.scoring_utils.clean_ipa_text` and split
    with :func:`~espnet3.systems.pr.metrics.scoring_utils.segment_ipa`, so a
    reference written as space-separated phones and a hypothesis written as one
    concatenated string are scored on the same units.

    Select it from ``conf/metrics.yaml`` with::

        metrics:
          - metric:
              _target_: espnet3.systems.pr.metrics.per.PER
    """

    def __init__(self, ref_key: str = "ref", hyp_key: str = "hyp") -> None:
        """Initialize the PER metric.

        Args:
            ref_key: Key name for reference transcript entries, resolved to
                ``<inference_dir>/<test_name>/<ref_key>.scp``.
            hyp_key: Key name for hypothesis transcript entries.
        """
        self.ref_key = ref_key
        self.hyp_key = hyp_key

    def _tokenize(self, text: str) -> str:
        """Normalize, split into phones, and join with spaces.

        Args:
            text: Raw IPA transcript.

        Returns:
            Space-joined phones, or a placeholder when nothing is recognizable,
            because jiwer rejects empty references.
        """
        phones = segment_ipa(clean_ipa_text(text))
        return " ".join(phones) if phones else "."

    def _ensure_jiwer(self) -> None:
        """Raise an error if the optional jiwer dependency is missing.

        Raises:
            RuntimeError: If ``jiwer`` is not installed.
        """
        if jiwer is None:
            raise RuntimeError(
                "jiwer is required to compute PER. "
                "Please install it with `pip install espnet[asr]`."
            )

    def __call__(
        self,
        data: Dict[str, Path],
        test_name: str,
        inference_dir: Path,
    ) -> Dict[str, float]:
        """Compute PER, write the alignment, and return the score.

        Args:
            data: Mapping with ``data[self.ref_key]`` and ``data[self.hyp_key]``
                SCP files, aligned by utterance ID.
            test_name: Test set name, used as the output subdirectory.
            inference_dir: Base directory the alignment file is written under.

        Returns:
            ``{"PER": <percentage>}``.

        Raises:
            RuntimeError: If ``jiwer`` or ``panphon`` is not installed.
            AssertionError: If the two SCP files are not aligned by utterance ID.
        """
        self._ensure_jiwer()
        refs = []
        hyps = []
        for _, row in self.iter_inputs(data, self.ref_key, self.hyp_key):
            refs.append(self._tokenize(row[self.ref_key]))
            hyps.append(self._tokenize(row[self.hyp_key]))

        score = jiwer.wer(refs, hyps) * 100
        details = jiwer.process_words(refs, hyps)

        test_dir = Path(inference_dir) / test_name
        test_dir.mkdir(parents=True, exist_ok=True)
        with (test_dir / "per_alignment").open("w", encoding="utf-8") as f:
            f.write(jiwer.visualize_alignment(details))

        return {"PER": round(score, 1)}
