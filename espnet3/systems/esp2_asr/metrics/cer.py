"""Character error rate metric utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

try:
    import jiwer
except ImportError:
    jiwer = None

from espnet2.text.cleaner import TextCleaner
from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.contract import Field


class CER(BaseMetric):
    """Compute CER for a dataset.

    Reads ``data["ref"]``/``data["hyp"]``, the declared names; the
    metrics config's ``inputs:`` binds them to a source (see
    :class:`~espnet3.contract.metrics.check_metric_inputs`).

    Examples:
        >>> [f.name for f in CER().inputs]
        ['ref', 'hyp']
    """

    inputs = (Field("ref", "text"), Field("hyp", "text"))
    outputs = (Field("CER", "number"),)

    def __init__(self, clean_types: Iterable[str] | None = None) -> None:
        """Initialize the CER metric.

        Args:
            clean_types: Optional cleaner types passed to TextCleaner.
        """
        self.cleaner = TextCleaner(clean_types)
        super().__init__()
        self.add_state("hits", 0, "sum")
        self.add_state("substitutions", 0, "sum")
        self.add_state("deletions", 0, "sum")
        self.add_state("insertions", 0, "sum")
        self._alignments: List[str] = []

    def reset(self) -> None:
        """Reset the counts and drop alignment text from an unfinished run.

        Examples:
            >>> metric = CER()
            >>> metric.reset()
        """
        super().reset()
        self._alignments = []

    def _clean(self, text: str) -> str:
        """Clean text and provide a placeholder for empty strings.

        Args:
            text: Input text to clean.

        Returns:
            Cleaned string, or a placeholder to avoid empty inputs.
        """
        cleaned = self.cleaner(text).strip()
        return cleaned if cleaned else "."

    def _ensure_jiwer(self) -> None:
        """Raise an error if the optional jiwer dependency is missing.

        Raises:
            RuntimeError: If ``jiwer`` is not installed.
        """
        if jiwer is None:
            raise RuntimeError(
                "jiwer is required to compute CER. "
                "Please install it with `pip install espnet[asr]`."
            )

    def update(self, *, ref: Sequence[str], hyp: Sequence[str]) -> None:
        """Add the character error counts of one batch to the state.

        Args:
            ref: Reference texts, one per utterance.
            hyp: Hypothesis texts, aligned with ``ref``.

        Raises:
            RuntimeError: If ``jiwer`` is not installed.

        Examples:
            >>> metric = CER()
            >>> metric.update(ref=["abc"], hyp=["axc"])
        """
        self._ensure_jiwer()
        if not ref:
            return
        out = jiwer.process_characters(
            [self._clean(r) for r in ref], [self._clean(h) for h in hyp]
        )
        self.hits += out.hits
        self.substitutions += out.substitutions
        self.deletions += out.deletions
        self.insertions += out.insertions
        if self.collect_artifacts:
            self._alignments.append(jiwer.visualize_alignment(out, show_measures=False))

    def compute(self) -> Dict[str, float]:
        """Return the CER of everything added so far, from the summed counts.

        Returns:
            Dict[str, float]: ``{"CER": <percentage>}``, rounded to 2 places.

        Raises:
            ValueError: If the references have no characters.

        Examples:
            >>> metric = CER()
            >>> metric.update(ref=["abc"], hyp=["axc"])
            >>> metric.compute()
            {'CER': 33.33}
        """
        n = self.hits + self.substitutions + self.deletions
        if n == 0:
            raise ValueError("nothing to score: the references have no characters")
        errors = self.substitutions + self.deletions + self.insertions
        return {"CER": round(100 * errors / n, 2)}

    def write_artifacts(self, test_dir: Path) -> None:
        """Write the alignments kept by this ``__call__`` to ``test_dir/cer_alignment``.

        The file holds only the rows seen by the ``__call__`` that wrote it,
        then one summary line from the summed counts; the kept text is cleared
        afterwards.

        Args:
            test_dir: The test set's output directory, already created.

        Examples:
            >>> metric.write_artifacts(Path("infer/test-clean"))
        """
        n = self.hits + self.substitutions + self.deletions
        errors = self.substitutions + self.deletions + self.insertions
        rate = 100 * errors / n if n else 0.0
        summary = (
            f"\nCER: {rate:.2f} (hits={self.hits}, "
            f"substitutions={self.substitutions}, deletions={self.deletions}, "
            f"insertions={self.insertions})\n"
        )
        (test_dir / "cer_alignment").write_text(
            "".join(self._alignments) + summary, encoding="utf-8"
        )
        self._alignments = []
