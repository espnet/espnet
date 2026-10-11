r"""Shared error rate metric for ASR.

This module provides the ErrorRateMetric class, which implements WER, CER, and TER
on top of the BaseMetric accumulator.
"""

from __future__ import annotations

from pathlib import Path
from typing import (
    ClassVar,
    Dict,
    Iterable,
    Literal,
    Sequence,
)

try:
    import jiwer
except ImportError:
    jiwer = None

from espnet2.text.cleaner import TextCleaner
from espnet2.text.sentencepiece_tokenizer import SentencepiecesTokenizer
from espnet3.api.inference import Field
from espnet3.components.metrics.base_metric import BaseMetric


class ErrorRateMetric(BaseMetric):
    """Shared error rate metric.

    Implements WER, CER, or TER by accumulating jiwer counts.

    Examples:
        >>> from espnet3.systems.esp2_asr.metrics.wer import WER
        >>> metric = WER()
        >>> metric.update(ref=["a b c"], hyp=["a x c"])
        >>> metric.compute()
        {'WER': 33.33}
    """

    inputs = (Field("ref", "text"), Field("hyp", "text"))
    output_key: ClassVar[str]  # "WER" | "CER" | "TER"
    level: ClassVar[Literal["word", "char"]]  # "word" for WER, "char" for CER
    artifact_name: ClassVar[str]  # "wer_alignment" | "cer_alignment" | "ter_alignment"

    def __init__(
        self,
        clean_types: Iterable[str] | None = None,
        bpemodel: str | Path | None = None,
    ) -> None:
        """Initialize the ErrorRateMetric.

        Args:
            clean_types: Optional cleaner types passed to TextCleaner.
            bpemodel: Optional SentencePiece model for TER.
        """
        self.cleaner = TextCleaner(clean_types)
        if bpemodel is not None:
            self.tokenizer = SentencepiecesTokenizer(bpemodel)
        else:
            self.tokenizer = None
        super().__init__()
        self.add_state("hits", 0, "sum")
        self.add_state("substitutions", 0, "sum")
        self.add_state("deletions", 0, "sum")
        self.add_state("insertions", 0, "sum")
        self.add_state("sentences", 0, "sum")
        self.add_state("alignments", [], "cat")

    def _prepare(self, text: str) -> str:
        """Clean text and provide a placeholder for empty strings.

        Args:
            text: Input text.

        Returns:
            Cleaned string, or a placeholder to avoid empty inputs.

        Examples:
            >>> metric._prepare("  a  ")
            'a'
        """
        cleaned = self.cleaner(text).strip()
        if not cleaned:
            return "."

        if self.tokenizer is not None:
            # TER mode
            tokens = self.tokenizer.text2tokens(cleaned)
            return " ".join(tokens) if tokens else "."
        else:
            # WER/CER mode
            return cleaned

    def _ensure_jiwer(self) -> None:
        """Raise if jiwer is missing.

        Examples:
            >>> metric._ensure_jiwer()
        """
        if jiwer is None:
            raise RuntimeError(
                "jiwer is required to compute error rates. "
                "Please install it with `pip install espnet[asr]`."
            )

    def update(self, *, ref: Sequence[str], hyp: Sequence[str]) -> None:
        """Accumulate jiwer counts.

        Args:
            ref: Reference sequences.
            hyp: Hypothesis sequences.

        Examples:
            >>> metric.update(ref=["a b"], hyp=["a c"])
        """
        self._ensure_jiwer()
        if not ref:
            return

        process = (
            jiwer.process_characters if self.level == "char" else jiwer.process_words
        )
        out = process([self._prepare(r) for r in ref], [self._prepare(h) for h in hyp])

        self.hits += out.hits
        self.substitutions += out.substitutions
        self.deletions += out.deletions
        self.insertions += out.insertions
        self.sentences += len(ref)
        self.alignments.append(jiwer.visualize_alignment(out, show_measures=False))

    def compute(self) -> Dict[str, float]:
        """Compute error rate.

        Returns:
            Dict containing the error rate.

        Examples:
            >>> metric.compute()
            {'WER': 33.33}
        """
        n = self.hits + self.substitutions + self.deletions
        if n == 0:
            raise ValueError("nothing to score: the references have no tokens")

        score = 100 * (self.substitutions + self.deletions + self.insertions) / n
        return {self.output_key: round(score, 2)}

    def write_artifacts(self, test_dir: Path) -> None:
        """Write alignment and summary to the artifacts directory.

        Args:
            test_dir (Path): Directory for alignment outputs.

        Examples:
            >>> metric.write_artifacts(Path("/tmp"))
        """
        n = self.hits + self.substitutions + self.deletions
        errors = self.substitutions + self.deletions + self.insertions
        rate = 100 * errors / n if n else 0.0
        summary = (
            f"\nSummary: sentences={self.sentences}, hits={self.hits}, "
            f"substitutions={self.substitutions}, deletions={self.deletions}, "
            f"insertions={self.insertions}, rate={rate:.2f}\n"
        )
        (test_dir / self.artifact_name).write_text(
            "".join(self.alignments) + summary, encoding="utf-8"
        )
