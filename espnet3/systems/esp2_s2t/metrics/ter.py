"""Token error rate over BPE pieces, ``s2t.sh``'s ``--token_type bpe`` pass."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

try:
    import jiwer
except ImportError:
    jiwer = None

from espnet2.text.cleaner import TextCleaner
from espnet2.text.sentencepiece_tokenizer import SentencepiecesTokenizer
from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.contract import Field
from espnet3.systems.esp2_s2t.src.tags import configured_pattern, strip_tags


class TER(BaseMetric):
    """Compute TER over every row, tags removed from both sides.

    Not sacreBLEU's TER (translation edit rate), which
    ``espnet3/systems/esp2_st`` reports under the same name. s2t.sh's ``ter``
    is an error rate over the model's own subword units: both sides are
    tokenized with the recipe's SentencePiece model and then scored like WER.

    s2t.sh cannot remove non-linguistic symbols in its BPE pass even in
    principle -- ``build_tokenizer.py:38-40`` raises for ``token_type=bpe`` --
    so its reference keeps the tags, which tokenize as single pieces because
    they are the tokenizer's ``user_defined_symbols``. Here they are stripped
    before tokenizing, as for the other two rates.

    Reads ``data["ref"]``/``data["hyp"]``, the declared names; the metrics
    config's ``inputs:`` binds them to a source (see
    :func:`~espnet3.contract.metrics.check_metric_inputs`).

    Examples:
        >>> [f.name for f in TER.inputs]
        ['ref', 'hyp']
    """

    inputs = (Field("ref", "text"), Field("hyp", "text"))
    outputs = (Field("TER", "number"),)

    def __init__(
        self,
        bpemodel: str | Path,
        nlsyms: Any = None,
        ref_cleaner: Iterable[str] | None = None,
        hyp_cleaner: Iterable[str] | None = None,
        remove_tags: bool = True,
    ) -> None:
        """Initialize the TER metric.

        Args:
            bpemodel: Path to the SentencePiece model the recipe trained; it
                has to match the training config's ``tokenizer.save_path`` and
                ``model_type``.
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
        self.tokenizer = SentencepiecesTokenizer(bpemodel)
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

    def _tokenize(self, text: str, cleaner: TextCleaner) -> str:
        """Strip the tags, clean, and return the pieces space-joined for jiwer."""
        if self.pattern is not None:
            text = strip_tags(text, self.pattern)
        cleaned = cleaner(text).strip()
        if not cleaned:
            return "."
        tokens = self.tokenizer.text2tokens(cleaned)
        return " ".join(tokens) if tokens else "."

    def _ensure_jiwer(self) -> None:
        """Raise with an install hint rather than an ImportError at scoring time."""
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
            >>> metric = TER(bpemodel="bpe.model", remove_tags=False)
            >>> metric.update(ref=["a b c"], hyp=["a x c"])
        """
        self._ensure_jiwer()
        if not ref:
            return
        out = jiwer.process_words(
            [self._tokenize(r, self.ref_cleaner) for r in ref],
            [self._tokenize(h, self.hyp_cleaner) for h in hyp],
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
            >>> metric = TER(bpemodel="bpe.model", remove_tags=False)
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
