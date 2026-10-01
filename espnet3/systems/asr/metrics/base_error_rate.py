"""Shared base for the error-rate metrics, which are adapters over SPLET.

The metrics themselves live in `splet` -- `wer`, `cer` and `token_error_rate`
are entries in its `METRIC_CHOICES` table -- and the scoring, the pooling and
the alignment rendering are all SPLET's. What is left here is the part that is
ESPnet's, and only that: reading the SCP files `measure` hands over, naming
espnet2's text cleaners, writing the alignment file beside the hypotheses, and
reporting under the keys `metrics.json` already uses.

Two behaviours changed when this stopped calling `jiwer`, both deliberate.
Neither is invisible, so both are stated here with what they move:

* **Case.** `sclite` compares case-insensitively unless it is given `-s`, and
  only four egs2 corpora pass it, so `case="fold"` is the default here as it
  is in SPLET. `jiwer` was case-sensitive. This matters whenever *either* side
  carries case, not only the reference: a model that emits cased text against
  a lower-case reference was being charged a substitution per word for it. On
  `egs3/owsm_v4`'s MLS_en_test that is the difference between 28.95 and 23.05
  WER. The four spgispeech runs, lower-case on both sides, do not move at all.
  Pass `case="sensitive"` for the old comparison.
* **Empty hypotheses.** The previous code substituted `"."` for an empty
  string on both sides. An undecodable utterance then scored one substitution
  instead of a deletion per reference word, which flatters the system; #6735
  removed the same placeholder from the BLEU metric for the same reason. The
  reference length is now the reference's own length. No egs3 ASR output has
  an empty hypothesis today, so nothing moves on what is in the repository.

`sclite`'s totals are an upper bound on `jiwer`'s rather than equal to them,
because it minimises a weighted cost rather than an edit count. The two agree
on clean output and separate as the error rate rises. See
:mod:`splet.alignment`.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from espnet3.components.metrics.base_metric import BaseMetric
from splet import list_scoring, load_score_modules, load_summary

# espnet2's TextCleaner names, mapped onto SPLET's normalizers. Only the two
# Whisper ones can change an egs2 ASR score -- they are the only cleaners any
# ASR-family recipe sets -- and they are the only ones SPLET implements.
_CLEANER_TO_NORMALIZER = {
    "whisper_en": {"name": "whisper", "language": "en"},
    "whisper_basic": {"name": "whisper", "language": "basic"},
}

#: Counts SPLET reports beside a rate, as suffixes on the metric's name.
_COUNT_SUFFIXES = ("errors", "ref_len", "sub", "del", "ins", "hit")


def normalize_config(clean_types: Optional[Iterable[str]]) -> Optional[list]:
    """Translate espnet2 cleaner names into a SPLET normalize pipeline.

    Args:
        clean_types: Cleaner names as ``asr.sh`` would pass them, or None.

    Returns:
        A SPLET normalize config, or None when nothing was asked for.

    Raises:
        NotImplementedError: For a cleaner SPLET does not implement. Silently
            skipping it would report a score under a normalization that never
            ran.
    """
    names = list(clean_types or [])
    unsupported = [name for name in names if name not in _CLEANER_TO_NORMALIZER]
    if unsupported:
        raise NotImplementedError(
            f"SPLET does not implement the {unsupported} text cleaner(s). "
            f"Available: {sorted(_CLEANER_TO_NORMALIZER)}. These are the only "
            "cleaners any egs2 ASR-family recipe sets, so the others have not "
            "been ported; see espnet/espnet#6760."
        )
    return [_CLEANER_TO_NORMALIZER[name] for name in names] or None


class BaseErrorRate(BaseMetric):
    """An error rate computed by one of SPLET's utterance-tier metrics.

    Subclasses name the SPLET metric and the files they write; anything about
    how the rate is computed belongs in `splet`, not here.
    """

    #: Key this metric reports under in metrics.json.
    metric_name: str = ""
    #: Entry in SPLET's METRIC_CHOICES that does the work.
    splet_metric: str = ""
    #: File the per-utterance alignments are written to.
    alignment_filename: str = ""

    def __init__(
        self,
        ref_key: str = "ref",
        hyp_key: str = "hyp",
        clean_types: Optional[Iterable[str]] = None,
        case: str = "fold",
        costs: str = "sclite",
    ) -> None:
        """Initialize the metric.

        Args:
            ref_key: Key name for reference text entries.
            hyp_key: Key name for hypothesis text entries.
            clean_types: Cleaner names, as espnet2's TextCleaner takes them.
            case: ``"fold"`` compares case-insensitively, as ``sclite`` does
                by default. ``"sensitive"`` is ``sclite``'s ``-s``, and is
                what ``jiwer`` did.
            costs: ``"sclite"`` reproduces sclite's weighted alignment;
                ``"unit"`` is plain Levenshtein, which is what ``jiwer``
                computes.
        """
        self.ref_key = ref_key
        self.hyp_key = hyp_key
        self.clean_types = list(clean_types or [])
        self.case = case
        self.costs = costs

    @abstractmethod
    def tokenizer_conf(self) -> Dict[str, Any]:
        """Return the tokenizer options for this metric's SPLET entry."""
        raise NotImplementedError

    def score_config(self) -> List[Dict[str, Any]]:
        """Build the SPLET score config this metric scores with."""
        return [
            {
                "name": self.splet_metric,
                "tokenizer_conf": self.tokenizer_conf(),
                "normalize": normalize_config(self.clean_types),
                "case": self.case,
                "costs": self.costs,
                "keep_alignment": True,
            }
        ]

    def __call__(
        self,
        data: Dict[str, Path],
        test_name: str,
        inference_dir: Path,
    ) -> Dict[str, Any]:
        """Score a test set, write its alignments, and return the counts.

        Args:
            data: Mapping of input aliases to SCP paths. ``data[ref_key]`` and
                ``data[hyp_key]`` must share utterance IDs in the same order.
            test_name: Test set name, used for the output directory.
            inference_dir: Directory the test set's outputs live in.

        Returns:
            The rate as a percentage under :attr:`metric_name`, plus the
            counts it was pooled from: ``_errors``, ``_ref_len``, ``_sub``,
            ``_del``, ``_ins`` and ``_hit``. The rate is
            ``sum(errors) / sum(ref_len)``, which is what SCTK reports and is
            not the mean of the per-utterance rates.
        """
        references, hypotheses = {}, {}
        for utt_id, row in self.iter_inputs(data, self.ref_key, self.hyp_key):
            references[utt_id] = row[self.ref_key]
            hypotheses[utt_id] = row[self.hyp_key]

        modules = load_score_modules(self.score_config())
        score_info = list_scoring(hypotheses, modules, references)
        summary = load_summary(score_info)

        self._write_alignments(score_info, test_name, inference_dir)

        prefix = self.splet_metric
        result: Dict[str, Any] = {self.metric_name: round(100 * summary[prefix], 2)}
        for suffix in _COUNT_SUFFIXES:
            result[f"{self.metric_name}_{suffix}"] = summary[f"{prefix}_{suffix}"]
        return result

    def _write_alignments(self, score_info, test_name, inference_dir) -> None:
        """Write the alignments of the utterances that had an error.

        jiwer's visualize_alignment skipped the correct ones and this keeps
        that: on a 4000-utterance set rendering them all made the file eight
        times larger with nothing more to look at.
        """
        prefix = self.splet_metric
        rendered = [
            f"{score['key']}\n{score['alignment']}\n"
            for score in score_info
            if score[f"{prefix}_errors"]
        ]
        test_dir = Path(inference_dir) / test_name
        test_dir.mkdir(parents=True, exist_ok=True)
        with (test_dir / self.alignment_filename).open("w", encoding="utf-8") as f:
            f.write("\n".join(rendered))
