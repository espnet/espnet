"""BLEU metric for speech translation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

try:
    import sacrebleu
except ImportError:
    sacrebleu = None

from espnet2.text.cleaner import TextCleaner
from espnet3.components.metrics.base_metric import BaseMetric


class BLEU(BaseMetric):
    """Compute corpus BLEU for translation hypotheses.

    Mirrors ``espnet3.systems.asr.metrics.wer.WER`` in shape: it reads the
    ``ref``/``hyp`` SCP files that inference wrote, scores them, and writes a
    human-readable detail file next to them.

    sacreBLEU is used rather than ``nltk``'s sentence BLEU because it is what
    egs2's ``st.sh`` reports (``scripts/utils/score_bleu.sh`` calls
    ``sacrebleu``), and because corpus BLEU is not the mean of sentence BLEUs --
    scoring per sentence and averaging gives a different, lower number that is
    not comparable to published results.
    """

    def __init__(
        self,
        ref_key: str = "ref",
        hyp_key: str = "hyp",
        clean_types: Iterable[str] | None = None,
        tokenize: str = "13a",
        also_lowercase: bool = True,
    ) -> None:
        """Initialize the BLEU metric.

        Args:
            ref_key: Key name for reference text entries.
            hyp_key: Key name for hypothesis text entries.
            clean_types: Optional cleaner types passed to TextCleaner.
            tokenize: sacreBLEU tokenizer. ``13a`` is sacreBLEU's default and
                the one egs2 scores with; use ``zh``/``ja-mecab`` for those
                target languages, where whitespace is not a word boundary.
            also_lowercase: Additionally report case-insensitive BLEU as
                ``BLEU_lc``. egs2's st.sh reports both -- case-sensitive at
                st.sh:1604 and ``sacrebleu -lc`` at st.sh:1616 -- and papers
                quote the case-sensitive number. Both are produced by ONE
                instance because espnet3 keys results by metric CLASS PATH
                (espnet3/systems/base/metric.py:109), so two BLEU entries in
                metrics.yaml would overwrite each other.
        """
        self.cleaner = TextCleaner(clean_types)
        self.ref_key = ref_key
        self.hyp_key = hyp_key
        self.tokenize = tokenize
        self.also_lowercase = also_lowercase

    def _clean(self, text: str) -> str:
        """Clean text, mapping empty strings to a placeholder."""
        cleaned = self.cleaner(text).strip()
        return cleaned if cleaned else "."

    def _ensure_sacrebleu(self) -> None:
        """Raise if sacrebleu is missing or too old for the scorer API.

        ``sacrebleu.BLEU`` is a scorer class only from 2.0; in 1.x the same
        name is the score namedtuple, so the ``tokenize``/``lowercase``
        arguments below would fail confusingly. espnet declares
        ``sacrebleu>=1.5.1``, so the version is checked rather than assumed.
        """
        if sacrebleu is None:
            raise RuntimeError(
                "sacrebleu is required to compute BLEU. "
                "Please install it with `pip install 'sacrebleu>=2.0.0'`."
            )
        if not hasattr(sacrebleu, "BLEU") or not isinstance(sacrebleu.BLEU, type):
            raise RuntimeError(
                "BLEU needs sacrebleu >= 2.0.0 for its scorer API "
                f"(found {getattr(sacrebleu, '__version__', 'unknown')}). "
                "Please upgrade with `pip install -U 'sacrebleu>=2.0.0'`."
            )

    def __call__(
        self,
        data: Dict[str, Path],
        test_name: str,
        inference_dir: Path,
    ) -> Dict[str, float]:
        """Compute corpus BLEU, write the signature and detail, return the score.

        Args:
            data: Mapping of metric input aliases to SCP paths; expects
                ``data[self.ref_key]`` and ``data[self.hyp_key]``, aligned by
                utterance id.
            test_name: Test set name, used for the output subdirectory.
            inference_dir: Base directory inference wrote into.

        Returns:
            ``{"BLEU": <score>}``, plus the n-gram precisions and brevity
            penalty, which are what make one BLEU comparable to another.

        Raises:
            RuntimeError: If ``sacrebleu`` is not installed.
        """
        self._ensure_sacrebleu()
        refs, hyps = [], []
        for _, row in self.iter_inputs(data, self.ref_key, self.hyp_key):
            refs.append(self._clean(row[self.ref_key]))
            hyps.append(self._clean(row[self.hyp_key]))

        # Object API rather than the corpus_bleu() shortcut: it is the only way
        # to obtain the signature, and sacrebleu 2.x removed BLEUScore._signature.
        def score(lowercase: bool):
            scorer = sacrebleu.BLEU(tokenize=self.tokenize, lowercase=lowercase)
            return scorer.corpus_score(hyps, [refs]), str(scorer.get_signature())

        result, signature = score(lowercase=False)
        out = {
            "BLEU": round(result.score, 2),
            "BLEU_bp": round(result.bp, 4),
            **{
                f"BLEU_p{n}": round(p, 2)
                for n, p in enumerate(result.precisions, start=1)
            },
        }
        lines = [result.format(signature=signature)]

        if self.also_lowercase:
            lc_result, lc_signature = score(lowercase=True)
            out["BLEU_lc"] = round(lc_result.score, 2)
            lines.append(lc_result.format(signature=lc_signature))

        test_dir = Path(inference_dir) / test_name
        test_dir.mkdir(parents=True, exist_ok=True)
        with (test_dir / "bleu_detail").open("w", encoding="utf-8") as f:
            # The signature records tokenizer/smoothing/version, without which
            # a BLEU number is not reproducible or comparable across papers.
            for line in lines:
                f.write(f"{line}\n")
            f.write(f"sentences: {len(hyps)}\n")
        return out
