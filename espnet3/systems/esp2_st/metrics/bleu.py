"""Translation metrics for speech translation, as egs2's st.sh reports them."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    import sacrebleu
except ImportError:
    sacrebleu = None

from espnet2.text.cleaner import TextCleaner
from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.contract import Field
from espnet3.systems.esp2_st.src.normalization import remove_punctuation


class BLEU(BaseMetric):
    """Score translation hypotheses the way ``egs2/TEMPLATE/st1/st.sh`` does.

    Mirrors ``espnet3.systems.esp2_asr.metrics.wer.WER`` in shape: it takes
    the declared ``ref``/``hyp`` texts, which the metrics config's ``inputs:``
    binds to a source, and writes a human-readable detail file beside the
    scores. Unlike WER's counts, the texts themselves are the state: corpus
    BLEU is not the mean of per-batch BLEUs, so ``compute`` scores them all
    at once.

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

    inputs = (Field("ref", "text"), Field("hyp", "text"))

    def __init__(
        self,
        clean_types: Iterable[str] | None = None,
        tokenize: str = "13a",
        also_lowercase: bool = True,
        lc_remove_punctuation: bool = True,
        chrf: bool = True,
        ter: bool = True,
    ) -> None:
        """Initialize the metric.

        Args:
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
        self.tokenize = tokenize
        self.also_lowercase = also_lowercase
        self.lc_remove_punctuation = lc_remove_punctuation
        self.chrf = chrf
        self.ter = ter
        self.outputs = self._declare_outputs()
        super().__init__()
        # Corpus BLEU is not the mean of per-batch BLEUs, so the texts
        # themselves are the state and `compute` scores them in one go.
        self.add_state("refs", [], "cat")
        self.add_state("hyps", [], "cat")
        self._lines: List[str] = []

    def _declare_outputs(self) -> Tuple[Field, ...]:
        """Name every key :meth:`compute` returns, for the chosen passes."""
        names: List[str] = []
        for suffix in ("",) + (("_lc",) if self.also_lowercase else ()):
            names.append(f"BLEU{suffix}")
            names.append(f"BLEU_brevity_penalty{suffix}")
            names += [f"BLEU_{n}gram_prec{suffix}" for n in range(1, 5)]
            if self.chrf:
                names.append(f"chrF2{suffix}")
            if self.ter:
                names.append(f"TER{suffix}")
        return tuple(Field(name, "number") for name in names)

    def reset(self) -> None:
        """Reset the texts and drop detail lines from an unfinished run."""
        super().reset()
        self._lines = []

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
        ``sacrebleu>=2.0.0`` under the ``st`` extra, but the ``egs2`` extra
        still permits 1.5.1 and an existing environment may predate either, so
        the version is checked rather than assumed.
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

    def update(self, *, ref: Sequence[str], hyp: Sequence[str]) -> None:
        """Keep one batch's cleaned texts for the corpus-level score.

        Args:
            ref: Reference translations, one per utterance.
            hyp: Hypotheses, aligned with ``ref``.

        Examples:
            >>> metric = BLEU()
            >>> metric.update(ref=["hallo welt"], hyp=["hallo welt"])
        """
        self.refs.extend(self._clean(text) for text in ref)
        self.hyps.extend(self._clean(text) for text in hyp)

    def compute(self) -> Dict[str, float]:
        """Score both passes over everything added so far.

        Returns:
            ``BLEU``/``chrF2``/``TER`` for the case-sensitive pass, plus the
            BLEU brevity penalty and n-gram precisions, and the same keys
            suffixed ``_lc`` for the case-insensitive pass.

        Raises:
            RuntimeError: If ``sacrebleu`` is not installed or is too old.
        """
        self._ensure_sacrebleu()
        refs, hyps = self.refs, self.hyps
        out, self._lines = self._score(hyps, refs, lowercase=False, suffix="")

        if self.also_lowercase:
            lc_hyps, lc_refs = hyps, refs
            if self.lc_remove_punctuation:
                lc_hyps = [remove_punctuation(t) for t in hyps]
                lc_refs = [remove_punctuation(t) for t in refs]
            lc_out, lc_lines = self._score(
                lc_hyps, lc_refs, lowercase=True, suffix="_lc"
            )
            out.update(lc_out)
            self._lines.extend(lc_lines)
        return out

    def write_artifacts(self, test_dir: Path) -> None:
        """Write the scored passes to ``test_dir/bleu_detail``.

        Args:
            test_dir: The test set's output directory, already created.
        """
        with (test_dir / "bleu_detail").open("w", encoding="utf-8") as f:
            # The signature records tokenizer/smoothing/version, without which
            # a BLEU number is not reproducible or comparable across papers.
            for line in self._lines:
                f.write(f"{line}\n")
            f.write(f"sentences: {len(self.hyps)}\n")
        self._lines = []
