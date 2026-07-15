"""Token error rate metric utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

try:
    import jiwer
except ImportError:
    jiwer = None

from espnet2.text.cleaner import TextCleaner
from espnet2.text.sentencepiece_tokenizer import SentencepiecesTokenizer
from espnet3.components.metrics.base_metric import BaseMetric


class TER(BaseMetric):
    """Compute TER (token error rate) for a dataset.

    TER is the error rate over the model's subword (BPE) tokens: the reference
    and hypothesis text are tokenized with a SentencePiece model, then scored
    like WER over the resulting token sequences. This mirrors espnet2's Stage 13
    scoring, which computes ``ter`` at the ``bpe`` token level.
    """

    def __init__(
        self,
        bpemodel: str | Path,
        ref_key: str = "ref",
        hyp_key: str = "hyp",
        clean_types: Iterable[str] | None = None,
    ) -> None:
        """Initialize the TER metric.

        Args:
            bpemodel: Path to the SentencePiece model used to tokenize text into
                subword tokens (typically the recipe's trained ``bpe.model``).
            ref_key: Key name for reference text entries.
            hyp_key: Key name for hypothesis text entries.
            clean_types: Optional cleaner types passed to TextCleaner.
        """
        self.tokenizer = SentencepiecesTokenizer(bpemodel)
        self.cleaner = TextCleaner(clean_types)
        self.ref_key = ref_key
        self.hyp_key = hyp_key

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
        return " ".join(self.tokenizer.text2tokens(cleaned))

    def _ensure_jiwer(self) -> None:
        """Raise if the optional jiwer dependency is missing."""
        if jiwer is None:
            raise RuntimeError(
                "jiwer is required to compute TER. "
                "Please install it with `pip install espnet[asr]`."
            )

    def __call__(
        self,
        data: Dict[str, Path],
        test_name: str,
        inference_dir: Path,
    ) -> Dict[str, float]:
        """Compute TER, write alignment details, and return the metric.

        Args:
            data (Dict[str, Path]): Mapping with ``data[self.ref_key]`` and
                ``data[self.hyp_key]`` SCP files aligned by utterance ID.
            test_name (str): Test set name used for output directory naming.
            inference_dir (Path): Base directory for alignment outputs.

        Returns:
            Dict[str, float]: ``{"TER": <percentage>}``

        Raises:
            RuntimeError: If ``jiwer`` is not installed.
            AssertionError: If the reference and hypothesis SCP files are not
                aligned by utterance ID.
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
        with (test_dir / "ter_alignment").open("w", encoding="utf-8") as f:
            f.write(jiwer.visualize_alignment(details))

        return {"TER": round(score, 2)}
