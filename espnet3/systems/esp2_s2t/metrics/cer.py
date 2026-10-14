"""Character error rate, ``s2t.sh``'s ``--token_type char`` pass."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

try:
    import jiwer
except ImportError:
    jiwer = None

from espnet2.text.cleaner import TextCleaner
from espnet3.api.inference import Field
from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.systems.esp2_s2t.metrics.tags import configured_pattern, strip_tags


class CER(BaseMetric):
    """Compute CER over every row, tags removed from both sides.

    A space is one of the characters counted, which is what espnet2's
    ``CharTokenizer`` means by its ``<space>`` token: dropping spaces instead
    would score "thequick fox" against "the quick fox" as a perfect match.

    Reads ``data["ref"]``/``data["hyp"]``, the declared names; the metrics
    config's ``inputs:`` binds them to a source (see
    :func:`~espnet3.contract.metrics.check_metric_inputs`).

    Examples:
        >>> [f.name for f in CER.inputs]
        ['ref', 'hyp']
    """

    inputs = (Field("ref", "text"), Field("hyp", "text"))
    outputs = (Field("CER", "number"),)

    def __init__(
        self,
        nlsyms: Any = None,
        ref_cleaner: Iterable[str] | None = None,
        hyp_cleaner: Iterable[str] | None = None,
        remove_tags: bool = True,
    ) -> None:
        """Initialize the CER metric.

        Args:
            nlsyms: The recipe's OWSM symbols -- the same hydra callable, file
                or list that ``tokenizer.nlsyms`` names. Required unless
                ``remove_tags`` is False.
            ref_cleaner: TextCleaner types for the reference, as s2t.sh's
                ``--cleaner``.
            hyp_cleaner: TextCleaner types for the hypothesis, as s2t.sh's
                ``--hyp_cleaner``. Separate because s2t.sh keeps them separate.
            remove_tags: Strip OWSM tags from both sides, so they are
                comparable. On by default; False reproduces s2t.sh, whose own
                removal does not work on OWSM text.
        """
        self.pattern = configured_pattern(nlsyms, remove_tags)
        self.ref_cleaner = TextCleaner(ref_cleaner)
        self.hyp_cleaner = TextCleaner(hyp_cleaner)
        super().__init__()
        self.add_state("hits", 0, "sum")
        self.add_state("substitutions", 0, "sum")
        self.add_state("deletions", 0, "sum")
        self.add_state("insertions", 0, "sum")
        self._alignments: List[str] = []

    def reset(self) -> None:
        """Reset the counts and drop alignment text from an unfinished run."""
        super().reset()
        self._alignments = []

    def _clean(self, text: str, cleaner: TextCleaner) -> str:
        """Strip the tags, clean, and give jiwer a placeholder for an empty line."""
        if self.pattern is not None:
            text = strip_tags(text, self.pattern)
        cleaned = cleaner(text).strip()
        return cleaned if cleaned else "."

    def _ensure_jiwer(self) -> None:
        """Raise with an install hint rather than an ImportError at scoring time."""
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
            >>> metric = CER(remove_tags=False)
            >>> metric.update(ref=["abc"], hyp=["axc"])
        """
        self._ensure_jiwer()
        if not ref:
            return
        out = jiwer.process_characters(
            [self._clean(r, self.ref_cleaner) for r in ref],
            [self._clean(h, self.hyp_cleaner) for h in hyp],
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
            >>> metric = CER(remove_tags=False)
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
