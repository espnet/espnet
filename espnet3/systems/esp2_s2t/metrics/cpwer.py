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
from typing import Dict, List, Tuple

import editdistance
import numpy as np
from scipy.optimize import linear_sum_assignment

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


def edit_counts(ref: List[str], hyp: List[str]) -> Tuple[int, int, int, int]:
    """Word-level Levenshtein with operation counts -> (cor, sub, del, ins)."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bp = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0], bp[i][0] = i, "d"
    for j in range(1, m + 1):
        dp[0][j], bp[0][j] = j, "i"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                best, op = dp[i - 1][j - 1], "c"
            else:
                best, op = dp[i - 1][j - 1] + 1, "s"
            if dp[i - 1][j] + 1 < best:
                best, op = dp[i - 1][j] + 1, "d"
            if dp[i][j - 1] + 1 < best:
                best, op = dp[i][j - 1] + 1, "i"
            dp[i][j], bp[i][j] = best, op
    cor = sub = dele = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        op = bp[i][j]
        if op == "c":
            cor, i, j = cor + 1, i - 1, j - 1
        elif op == "s":
            sub, i, j = sub + 1, i - 1, j - 1
        elif op == "d":
            dele, i = dele + 1, i - 1
        else:
            ins, j = ins + 1, j - 1
    return cor, sub, dele, ins


def group_cpwer(ref_blocks: List[str], hyp_blocks: List[str]) -> Dict[str, int]:
    """Optimal-permutation error counts for a single utterance group.

    Blocks are padded with empty speakers to a square cost matrix so that
    unmatched reference blocks become deletions and unmatched hypothesis blocks
    become insertions.
    """
    ref_w = [b.split() for b in ref_blocks]
    hyp_w = [b.split() for b in hyp_blocks]
    k = max(len(ref_w), len(hyp_w))
    if k == 0:
        return {"cor": 0, "sub": 0, "del": 0, "ins": 0, "ref_len": 0}
    ref_w += [[] for _ in range(k - len(ref_w))]
    hyp_w += [[] for _ in range(k - len(hyp_w))]

    cost = np.zeros((k, k), dtype=np.int64)
    for i in range(k):
        for j in range(k):
            cost[i, j] = editdistance.eval(ref_w[i], hyp_w[j])
    rows, cols = linear_sum_assignment(cost)

    acc = {"cor": 0, "sub": 0, "del": 0, "ins": 0, "ref_len": 0}
    for i, j in zip(rows, cols):
        cor, sub, dele, ins = edit_counts(ref_w[i], hyp_w[j])
        acc["cor"] += cor
        acc["sub"] += sub
        acc["del"] += dele
        acc["ins"] += ins
        acc["ref_len"] += len(ref_w[i])
    return acc


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
            acc = group_cpwer(ref_blocks, hyp_blocks)
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
