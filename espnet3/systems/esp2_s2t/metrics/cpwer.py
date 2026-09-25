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


def pct(num: float, den: float) -> float:
    return 100.0 * num / den if den else 0.0


def utt_cpwer(errors: int, ref_len: int):
    """Per-utterance cpWER, or ``None`` when the reference is empty.

    ``pct`` returns 0.0 for a zero denominator, which would report a group
    whose reference is empty and whose hypothesis is not as a perfect score.
    Those insertions are still counted in the aggregate; only this
    per-utterance figure is undefined, so it is reported as ``null``.
    """
    return pct(errors, ref_len) if ref_len else None


class UtteranceGroupCpWER(BaseMetric):
    """cpWER over one utterance group at a time, reported as ``ug_cpWER``.

    Each utterance group holds one text block per speaker, separated by
    ``speaker_change_symbol``. The blocks arrive in an arbitrary order, so
    the score is taken over the assignment of hypothesis blocks to reference
    blocks that minimizes the total word distance, and that assignment is
    found within the group. A session-level cpWER would find one assignment
    for a whole meeting and is a harder problem; the two are not comparable.

    Args:
        ref_key: Alias of the reference SCP input.
        hyp_key: Alias of the hypothesis SCP input.
        clean_types: TextCleaner pipeline.
        speaker_change_symbol: The symbol the model separates speakers
            with. Required: it belongs to the checkpoint, and the wrong one
            would leave every group as a single block instead of failing.
    """

    def __init__(
        self,
        speaker_change_symbol: str,
        ref_key: str = "ref",
        hyp_key: str = "hyp",
        clean_types=None,
    ) -> None:
        self.ref_key = ref_key
        self.hyp_key = hyp_key
        self.cleaner = TextCleaner(list(clean_types) if clean_types else None)
        self.speaker_change_symbol = speaker_change_symbol

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

    def __call__(self, data, test_name, output_dir):
        """Score one test set.

        Args:
            data: Alias to path mapping. Needs ``self.ref_key`` and
                ``self.hyp_key``, both SCP files with matching utterance ids
                in the same order.
            test_name: Test set name, used for the side-file directory.
            output_dir: Root the side files are written under.

        Returns:
            ``{"ug_cpWER": <percentage rounded to two decimals>}``.

        Raises:
            ValueError: When every reference in the test set is empty, so
                the aggregate has nothing to score against. Loud on purpose:
                a zero denominator would otherwise report a perfect 0.00%.
        """
        self._ensure_meeteval()
        total = {"cor": 0, "sub": 0, "del": 0, "ins": 0, "ref_len": 0}
        by_nspk = defaultdict(
            lambda: {"cor": 0, "sub": 0, "del": 0, "ins": 0, "ref_len": 0}
        )
        per_utt = {}
        for utt_id, row in self.iter_inputs(data, self.ref_key, self.hyp_key):
            ref_blocks = split_speakers(
                row[self.ref_key], self.cleaner, self.speaker_change_symbol
            )
            hyp_blocks = split_speakers(
                row[self.hyp_key], self.cleaner, self.speaker_change_symbol
            )
            acc = count_group_errors(ref_blocks, hyp_blocks)
            for key in total:
                total[key] += acc[key]
                by_nspk[len(ref_blocks)][key] += acc[key]
            errors = acc["sub"] + acc["del"] + acc["ins"]
            per_utt[utt_id] = {
                "cpwer": utt_cpwer(errors, acc["ref_len"]),
                "errors": errors,
                "ref_len": acc["ref_len"],
                "num_ref_speakers": len(ref_blocks),
                "num_hyp_speakers": len(hyp_blocks),
            }

        errors = total["sub"] + total["del"] + total["ins"]
        if total["ref_len"] == 0:
            raise ValueError(
                f"cpWER cannot be computed for test set {test_name!r}: every "
                "reference was empty (total ref_len is 0). pct() returns "
                "0.0 for a zero denominator, which would otherwise misreport "
                "a perfect score instead of failing. The likely cause is an "
                "output function that wrote an empty string for every "
                f"reference. Check that the {self.ref_key!r} input actually "
                "carries reference text."
            )
        score = pct(errors, total["ref_len"])

        test_dir = Path(output_dir) / test_name
        test_dir.mkdir(parents=True, exist_ok=True)
        with (test_dir / "cpwer_per_utt.json").open("w", encoding="utf-8") as f:
            json.dump(per_utt, f, indent=2)
        with (test_dir / "cpwer_by_num_speakers.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    str(n): {
                        **counts,
                        "cpwer": utt_cpwer(
                            counts["sub"] + counts["del"] + counts["ins"],
                            counts["ref_len"],
                        ),
                    }
                    for n, counts in sorted(by_nspk.items())
                },
                f,
                indent=2,
            )

        return {"ug_cpWER": round(score, 2)}
