"""Token error rate metric utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

try:
    import jiwer
except ImportError:
    jiwer = None

from espnet2.text.cleaner import TextCleaner
from espnet2.text.sentencepiece_tokenizer import SentencepiecesTokenizer
from espnet3.components.metrics.base_metric import BaseMetric


class TER(BaseMetric):
    """Compute token error rates from configured reference and hypothesis inputs."""

    inputs = {"ref": str, "hyp": str}
    outputs = ("TER",)

    def __init__(
        self,
        bpemodel: str | Path,
        clean_types: Iterable[str] | None = None,
    ) -> None:
        """Initialize the TER metric.

        Args:
            bpemodel: Path to the SentencePiece model used to tokenize text into
                subword tokens (typically the recipe's trained ``bpe.model``).
            clean_types: Optional cleaner types passed to TextCleaner.
        """
        self.tokenizer = SentencepiecesTokenizer(bpemodel)
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
            >>> metric = TER(bpemodel="bpe.model")
            >>> metric.reset()
        """
        super().reset()
        self._alignments = []

    def _tokenize(self, text: str) -> str:
        """Clean text, tokenize into subword pieces, and join with spaces.

        Args:
            text: Input text.

        Returns:
            Space-joined subword tokens, or a placeholder to avoid empty inputs.
        """
        cleaned = self.cleaner(text).strip()
        if not cleaned:
            return "."
        tokens = self.tokenizer.text2tokens(cleaned)
        return " ".join(tokens) if tokens else "."

    def _ensure_jiwer(self) -> None:
        """Raise if the optional jiwer dependency is missing."""
        if jiwer is None:
            raise RuntimeError(
                "jiwer is required to compute TER. "
                "Please install it with `pip install espnet[asr]`."
            )

    def update(self, *, ref: Sequence[str], hyp: Sequence[str]) -> None:
        """Add the token error counts of one batch to the state.

        Args:
            ref: Reference texts, one per utterance.
            hyp: Hypothesis texts, aligned with ``ref``.

        Raises:
            RuntimeError: If ``jiwer`` is not installed.

        Examples:
            >>> metric = TER(bpemodel="bpe.model")
            >>> metric.update(ref=["a b c"], hyp=["a x c"])
        """
        self._ensure_jiwer()
        if not ref:
            return
        out = jiwer.process_words(
            [self._tokenize(r) for r in ref], [self._tokenize(h) for h in hyp]
        )
        self.hits += out.hits
        self.substitutions += out.substitutions
        self.deletions += out.deletions
        self.insertions += out.insertions
        if self.collect_artifacts:
            self._alignments.append(jiwer.visualize_alignment(out, show_measures=False))

    def compute(self) -> Dict[str, float]:
        """Return the TER of everything added so far, from the summed counts.

        Returns:
            Dict[str, float]: ``{"TER": <percentage>}``, rounded to 2 places.

        Raises:
            ValueError: If the references have no tokens.

        Examples:
            >>> metric = TER(bpemodel="bpe.model")
            >>> metric.update(ref=["a b c"], hyp=["a x c"])
            >>> metric.compute()
            {'TER': 33.33}
        """
        n = self.hits + self.substitutions + self.deletions
        if n == 0:
            raise ValueError("nothing to score: the references have no tokens")
        errors = self.substitutions + self.deletions + self.insertions
        return {"TER": round(100 * errors / n, 2)}

    def write_artifacts(self, test_dir: Path) -> None:
        """Write the alignments kept by this ``__call__`` to ``test_dir/ter_alignment``.

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
            f"\nTER: {rate:.2f} (hits={self.hits}, "
            f"substitutions={self.substitutions}, deletions={self.deletions}, "
            f"insertions={self.insertions})\n"
        )
        (test_dir / "ter_alignment").write_text(
            "".join(self._alignments) + summary, encoding="utf-8"
        )
        self._alignments = []
