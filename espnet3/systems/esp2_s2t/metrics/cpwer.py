"""Utterance-group cpWER.

Concatenated minimum-permutation WER, in the form the CHiME-6 challenge
defined, scored **per utterance group rather than per session**. Each group
is scored independently with its own optimal speaker permutation: the
assignment (Hungarian algorithm) of hypothesis speaker blocks to reference
speaker blocks that minimizes the total word error. Counts are then
aggregated across groups.

This is not session-level cpWER, where one permutation has to serve a whole
meeting. Resolving the assignment inside each group is the easier problem,
so these numbers are not comparable with published session-level ones. The
class and the reported key both say ``utterance group`` for that reason.
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

try:
    from meeteval.wer import cp_word_error_rate
except ImportError:
    cp_word_error_rate = None

from espnet2.text.cleaner import TextCleaner
from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.contract import Field

# The speaker-change symbol belongs to the checkpoint, so it is a constructor
# argument rather than a constant: a metric cannot know which symbol a model
# was trained with. Text written with a different spelling has to be converted
# before it is scored here.
_SEP = "▁SPKCHANGE▁"  # internal marker unlikely to occur in text
_SPECIAL_RE = re.compile(r"<\|[^|]*\|>")  # Whisper special tokens, e.g. <|1.20|>


def strip_special_tokens(text: str) -> str:
    """Remove Whisper special tokens (timestamps, <|endoftext|>, ...)."""
    return re.sub(r"\s+", " ", _SPECIAL_RE.sub(" ", text)).strip()


def split_speakers(text: str, cleaner, speaker_change_symbol: str) -> List[str]:
    """Split SOT text into per-speaker word lists.

    The text is split on ``speaker_change_symbol``, Whisper special tokens
    are stripped, the normalizer is applied, and empty blocks are dropped.
    """
    text = text.replace(speaker_change_symbol, _SEP)
    blocks = []
    for chunk in text.split(_SEP):
        chunk = strip_special_tokens(chunk)
        if cleaner is not None:
            chunk = cleaner(chunk).strip()
        if chunk:
            blocks.append(chunk)
    return blocks


def count_group_errors(ref_blocks: List[str], hyp_blocks: List[str]) -> Dict[str, int]:
    """Return optimal-permutation error counts for a single utterance group.

    meeteval finds the assignment of hypothesis speaker blocks to reference
    speaker blocks that minimizes the total word error, and reports the
    operation counts under it. Unmatched blocks on either side are handled by
    meeteval itself: an extra reference speaker becomes deletions, an extra
    hypothesis speaker becomes insertions.

    Note:
        The operation split is not the only one a minimal alignment admits.
        Where a word swap can be scored either as one substitution or as one
        deletion plus one insertion, meeteval and a hand-rolled Levenshtein
        traceback can disagree while agreeing on the total. Only the totals
        feed the reported rate; the split reaches the side files.

    Args:
        ref_blocks: One reference text block per speaker.
        hyp_blocks: One hypothesis text block per speaker.

    Returns:
        ``{"cor", "sub", "del", "ins", "ref_len"}`` word counts.
    """
    if not ref_blocks and not hyp_blocks:
        return {"cor": 0, "sub": 0, "del": 0, "ins": 0, "ref_len": 0}

    result = cp_word_error_rate(
        list(ref_blocks) or [""],
        list(hyp_blocks) or [""],
        # The blocks carry no timestamps, and their order is the speaker
        # order this recipe serialized them in. Saying so keeps meeteval from
        # warning once per utterance that it cannot sort them.
        reference_sort=False,
        hypothesis_sort=False,
    )
    return {
        # ref_len = cor + sub + del, so the correct count follows from the
        # other two rather than being reported separately.
        "cor": result.length - result.substitutions - result.deletions,
        "sub": result.substitutions,
        "del": result.deletions,
        "ins": result.insertions,
        "ref_len": result.length,
    }


def get_percentage(numerator: float, denominator: float) -> float:
    """Return a percentage, or 0.0 when there is nothing to divide by."""
    return 100.0 * numerator / denominator if denominator else 0.0


def get_utt_cpwer(errors: int, ref_len: int):
    """Per-utterance cpWER, or ``None`` when the reference is empty.

    ``get_percentage`` returns 0.0 for a zero denominator, which would report
    a group whose reference is empty and whose hypothesis is not as a perfect
    score.
    Those insertions are still counted in the aggregate; only this
    per-utterance figure is undefined, so it is reported as ``null``.
    """
    return get_percentage(errors, ref_len) if ref_len else None


# The order of a per-group row in the report state.
_ROW = ("num_ref_speakers", "num_hyp_speakers", "cor", "sub", "del", "ins", "ref_len")


class UtteranceGroupCpWER(BaseMetric):
    """cpWER over one utterance group at a time, reported as ``ug_cpWER``.

    Each utterance group holds one text block per speaker, separated by
    ``speaker_change_symbol``. The blocks arrive in an arbitrary order, so
    the score is taken over the assignment of hypothesis blocks to reference
    blocks that minimizes the total word distance, and that assignment is
    found within the group. A session-level cpWER would find one assignment
    for a whole meeting and is a harder problem; the two are not comparable.

    Both inputs are SOT text. Timestamps and other Whisper special tokens are
    stripped before scoring, so the hypothesis can keep them for DER.

    Args:
        speaker_change_symbol: The symbol the model separates speakers
            with. Required: it belongs to the checkpoint, and the wrong one
            would leave every group as a single block instead of failing.
        clean_types: TextCleaner pipeline.

    Examples:
        >>> metric = UtteranceGroupCpWER("????")
        >>> metric.update(ref=["<|0.00|> a b ???? <|0.00|> c<|1.00|>"],
        ...               hyp=["<|0.00|> c ???? <|0.00|> a x<|1.00|>"])
        >>> metric.compute()
        {'ug_cpWER': 33.33}
    """

    inputs = (Field("ref", "text"), Field("hyp", "text"))
    outputs = (Field("ug_cpWER", "number"),)

    def __init__(self, speaker_change_symbol: str, clean_types=None) -> None:
        """Build the metric. See the class docstring for the arguments."""
        if not speaker_change_symbol:
            raise ValueError(
                "speaker_change_symbol is required and must be the symbol the "
                "checkpoint was trained with. Splitting on nothing scores a "
                "whole utterance group as one speaker, which reads as a poor "
                "result rather than as a misconfiguration."
            )
        self.cleaner = TextCleaner(list(clean_types) if clean_types else None)
        self.speaker_change_symbol = speaker_change_symbol
        super().__init__()
        self.add_state("errors", 0, "sum")
        self.add_state("ref_len", 0, "sum")
        # One _ROW per group, kept only while a report will be written.
        self.add_state("groups", [], "cat")

    def _ensure_meeteval(self) -> None:
        """Raise an error if the optional meeteval dependency is missing.

        Raises:
            RuntimeError: If ``meeteval`` is not installed.
        """
        if cp_word_error_rate is None:
            raise RuntimeError(
                "meeteval is required to compute cpWER. "
                "Please install it with `pip install espnet[s2t]`."
            )

    def __call__(self, data, test_name, output_dir=None):
        """Score one test set from its SCP files, as ``measure`` calls it.

        As ``BaseMetric.__call__``, keeping ``data`` so that
        :meth:`write_artifacts` can key its per-group report by utterance id,
        which :meth:`update` never sees.
        """
        self._data = data
        return super().__call__(data, test_name, output_dir)

    def update(self, *, ref, hyp) -> None:
        """Add the error counts of each utterance group in the batch."""
        self._ensure_meeteval()
        for reference, hypothesis in zip(ref, hyp):
            ref_blocks = split_speakers(
                reference, self.cleaner, self.speaker_change_symbol
            )
            hyp_blocks = split_speakers(
                hypothesis, self.cleaner, self.speaker_change_symbol
            )
            counts = count_group_errors(ref_blocks, hyp_blocks)
            self.errors += counts["sub"] + counts["del"] + counts["ins"]
            self.ref_len += counts["ref_len"]
            if self.collect_artifacts:
                counts["num_ref_speakers"] = len(ref_blocks)
                counts["num_hyp_speakers"] = len(hyp_blocks)
                self.groups.append([counts[key] for key in _ROW])

    def compute(self):
        """Return the summed errors over the summed reference words.

        Raises:
            ValueError: When every reference is empty, so the aggregate has
                nothing to score against. Loud on purpose: a zero denominator
                would otherwise report a perfect 0.00%.
        """
        if self.ref_len == 0:
            raise ValueError(
                "cpWER cannot be computed: every reference was empty (total "
                "ref_len is 0). Check that the ref input is bound to the "
                "reference text, for example dataset:text."
            )
        return {"ug_cpWER": round(get_percentage(self.errors, self.ref_len), 2)}

    def write_artifacts(self, test_dir: Path) -> None:
        """Write the per-group and by-speaker-count reports.

        Args:
            test_dir: Directory the two JSON files are written into.
        """
        utt_ids = [utt_id for utt_id, _ in self.iter_inputs(self._data, "hyp")]
        if len(utt_ids) != len(self.groups):
            raise RuntimeError(
                f"{len(self.groups)} scored groups for {len(utt_ids)} hypotheses"
            )
        per_utt = {}
        by_nspk = defaultdict(
            lambda: {"cor": 0, "sub": 0, "del": 0, "ins": 0, "ref_len": 0}
        )
        for utt_id, row in zip(utt_ids, self.groups):
            group = dict(zip(_ROW, row))
            errors = group["sub"] + group["del"] + group["ins"]
            per_utt[utt_id] = {
                "cpwer": get_utt_cpwer(errors, group["ref_len"]),
                "errors": errors,
                "ref_len": group["ref_len"],
                "num_ref_speakers": group["num_ref_speakers"],
                "num_hyp_speakers": group["num_hyp_speakers"],
            }
            for key in by_nspk[group["num_ref_speakers"]]:
                by_nspk[group["num_ref_speakers"]][key] += group[key]

        with (test_dir / "cpwer_per_utt.json").open("w", encoding="utf-8") as f:
            json.dump(per_utt, f, indent=2)
        with (test_dir / "cpwer_by_num_speakers.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    str(n): {
                        **counts,
                        "cpwer": get_utt_cpwer(
                            counts["sub"] + counts["del"] + counts["ins"],
                            counts["ref_len"],
                        ),
                    }
                    for n, counts in sorted(by_nspk.items())
                },
                f,
                indent=2,
            )
