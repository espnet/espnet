"""Translation metrics for speech translation, as egs2's st.sh reports them."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    import sacrebleu
except ImportError:
    sacrebleu = None

from espnet2.text.cleaner import TextCleaner
from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.systems.st.normalization import remove_punctuation


class BLEU(BaseMetric):
    """Score translation hypotheses the way ``egs2/TEMPLATE/st1/st.sh`` does.

    Mirrors ``espnet3.systems.asr.metrics.wer.WER`` in shape: it reads the
    ``ref``/``hyp`` SCP files that inference wrote, scores them, and writes a
    human-readable detail file next to them.

    sacreBLEU is used rather than ``nltk``'s sentence BLEU because it is what
    ``st.sh`` reports (``scripts/utils/score_bleu.sh`` calls ``sacrebleu``), and
    because corpus BLEU is not the mean of sentence BLEUs -- scoring per
    sentence and averaging gives a different, lower number that is not
    comparable to published results.

    **Correspondence with st.sh** (st.sh:1551-1620), which scores twice:

    * ``sacrebleu -m bleu chrf ter`` (st.sh:1604) -> ``BLEU``, ``chrF2``,
      ``TER``;
    * ``scripts/utils/remove_punctuation.pl`` (st.sh:1609-1612), then
      ``sacrebleu -lc -m bleu chrf ter`` (st.sh:1616) -> the same three keys
      suffixed ``_lc``.

    Note the asymmetry in sacreBLEU's own CLI, which is reproduced here: ``-lc``
    is ``--lowercase`` on BLEU *only* (``dest='bleu_lowercase'``). chrF keeps its
    separate ``--chrf-lowercase`` (off) and TER lowercases by default
    (``--ter-case-sensitive`` is off), so in st.sh's second pass chrF is
    case-SENSITIVE and TER is case-INsensitive in both passes. The ``_lc``
    results therefore differ from the first pass mainly by punctuation removal.

    **Detokenization.** st.sh word-tokenizes both sides and then runs Moses
    ``detokenizer.perl`` to undo it, because egs2's ST data prep stores Moses
    ``tokenizer.perl`` output (``Hello , world .``). A recipe that keeps natural
    text on both sides -- references straight from the corpus, hypotheses
    straight out of SentencePiece -- is already at the state st.sh detokenizes
    *to*, so the round trip is a no-op and is omitted rather than reimplemented.
    If a recipe does feed this metric Moses-tokenized text, detokenize it in the
    inference ``output_fn`` first; scoring tokenized against detokenized text
    silently inflates BLEU.
    """

    def __init__(
        self,
        ref_key: str = "ref",
        hyp_key: str = "hyp",
        clean_types: Iterable[str] | None = None,
        tokenize: str = "13a",
        also_lowercase: bool = True,
        lc_remove_punctuation: bool = True,
        chrf: bool = True,
        ter: bool = True,
    ) -> None:
        """Initialize the metric.

        Args:
            ref_key: Key name for reference text entries.
            hyp_key: Key name for hypothesis text entries.
            clean_types: Optional cleaner types passed to TextCleaner. st.sh
                applies its ``--cleaner`` to the reference only and deliberately
                not to the hypothesis (st.sh:1563, "Don't use cleaner for hyp");
                this applies it to neither by default, which is the same thing
                whenever no cleaner is configured.
            tokenize: sacreBLEU tokenizer. ``13a`` is sacreBLEU's default and
                the one egs2 scores with; use ``zh``/``ja-mecab`` for those
                target languages, where whitespace is not a word boundary.
            also_lowercase: Additionally report the case-insensitive pass,
                suffixed ``_lc``. st.sh reports both (case-sensitive at
                st.sh:1604, case-insensitive at st.sh:1616) and papers quote the
                case-sensitive number. Both passes come from ONE instance
                because espnet3 keys results by metric CLASS PATH
                (espnet3/systems/base/metric.py:109), so two BLEU entries in
                metrics.yaml would overwrite each other.
            lc_remove_punctuation: Strip punctuation before the case-insensitive
                pass, as st.sh does with ``scripts/utils/remove_punctuation.pl``
                (st.sh:1609-1612). Lowercasing alone is NOT what egs2 calls
                case-insensitive BLEU, and scores a point or so lower.
            chrf: Report chrF2, as ``st.sh``'s ``-m bleu chrf ter`` does.
            ter: Report TER (lower is better), likewise.
        """
        self.cleaner = TextCleaner(clean_types)
        self.ref_key = ref_key
        self.hyp_key = hyp_key
        self.tokenize = tokenize
        self.also_lowercase = also_lowercase
        self.lc_remove_punctuation = lc_remove_punctuation
        self.chrf = chrf
        self.ter = ter

    def _clean(self, text: str) -> str:
        """Clean and strip one line.

        An empty result is kept empty rather than replaced by a placeholder:
        st.sh scores whatever the decoder emitted, and substituting a token for
        an empty hypothesis hands it a free n-gram match. The recipes that hit
        this are the ones whose inference returns a blank string for an
        utterance too short to decode.
        """
        return self.cleaner(text).strip()

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

    def _scorers(self, lowercase: bool) -> List:
        """One pass of ``sacrebleu -m bleu chrf ter``, with st.sh's flags.

        Only BLEU takes ``lowercase``: that is what sacreBLEU's ``-lc`` sets
        (``dest='bleu_lowercase'``). chrF and TER keep their CLI defaults so the
        numbers match what st.sh printed.
        """
        # Object API rather than the corpus_bleu() shortcut: it is the only way
        # to obtain the signature, and sacrebleu 2.x removed BLEUScore._signature.
        scorers = [sacrebleu.BLEU(tokenize=self.tokenize, lowercase=lowercase)]
        if self.chrf:
            scorers.append(sacrebleu.CHRF())
        if self.ter:
            scorers.append(sacrebleu.TER())
        return scorers

    def _score(
        self, hyps: List[str], refs: List[str], lowercase: bool, suffix: str
    ) -> Tuple[Dict[str, float], List[str]]:
        """Score one pass, returning its metrics and its printable lines."""
        out: Dict[str, float] = {}
        lines: List[str] = []
        for scorer in self._scorers(lowercase):
            result = scorer.corpus_score(hyps, [refs])
            signature = str(scorer.get_signature())
            out[f"{result.name}{suffix}"] = round(result.score, 2)
            if isinstance(scorer, sacrebleu.BLEU):
                # The brevity penalty and n-gram precisions are what make one
                # BLEU comparable to another. Spelled out rather than as
                # sacreBLEU's "60.0/32.0/19.4/12.2" block or the literature's
                # p_n, because these become column headers in metrics.json.
                out[f"BLEU_brevity_penalty{suffix}"] = round(result.bp, 4)
                out.update(
                    {
                        f"BLEU_{n}gram_prec{suffix}": round(p, 2)
                        for n, p in enumerate(result.precisions, start=1)
                    }
                )
            lines.append(result.format(signature=signature))
        return out, lines

    def __call__(
        self,
        data: Dict[str, Path],
        test_name: str,
        inference_dir: Path,
    ) -> Dict[str, float]:
        """Score both passes, write the detail file, return the metrics.

        Args:
            data: Mapping of metric input aliases to SCP paths; expects
                ``data[self.ref_key]`` and ``data[self.hyp_key]``, aligned by
                utterance id.
            test_name: Test set name, used for the output subdirectory.
            inference_dir: Base directory inference wrote into.

        Returns:
            ``BLEU``/``chrF2``/``TER`` for the case-sensitive pass, plus the
            BLEU brevity penalty and n-gram precisions, and the same keys
            suffixed ``_lc`` for the case-insensitive pass.

        Raises:
            RuntimeError: If ``sacrebleu`` is not installed or is too old.
        """
        self._ensure_sacrebleu()
        refs, hyps = [], []
        for _, row in self.iter_inputs(data, self.ref_key, self.hyp_key):
            refs.append(self._clean(row[self.ref_key]))
            hyps.append(self._clean(row[self.hyp_key]))

        out, lines = self._score(hyps, refs, lowercase=False, suffix="")

        if self.also_lowercase:
            lc_hyps, lc_refs = hyps, refs
            if self.lc_remove_punctuation:
                lc_hyps = [remove_punctuation(t) for t in hyps]
                lc_refs = [remove_punctuation(t) for t in refs]
            lc_out, lc_lines = self._score(
                lc_hyps, lc_refs, lowercase=True, suffix="_lc"
            )
            out.update(lc_out)
            lines.extend(lc_lines)

        test_dir = Path(inference_dir) / test_name
        test_dir.mkdir(parents=True, exist_ok=True)
        with (test_dir / "bleu_detail").open("w", encoding="utf-8") as f:
            # The signature records tokenizer/smoothing/version, without which
            # a BLEU number is not reproducible or comparable across papers.
            for line in lines:
                f.write(f"{line}\n")
            f.write(f"sentences: {len(hyps)}\n")
        return out
