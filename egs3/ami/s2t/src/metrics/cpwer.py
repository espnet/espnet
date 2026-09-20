"""cpWER metric for the AMI SOT recipe.

Concatenated minimum-permutation WER, in the standard form the CHiME-6
challenge defined, scored per utterance group rather than per session. Each
group is scored independently with its own optimal speaker permutation: the
assignment (Hungarian algorithm) of hypothesis speaker blocks to reference
speaker blocks that minimizes the total word error. Counts are then
aggregated across all groups to give the utterance-group cpWER. This is not
session-level cpWER, where a single permutation is found for a whole
meeting.

``split_speakers``, ``edit_counts`` and ``group_cpwer`` are a port of the
scorer used by the ESPnet2 AMI SOT recipe. What the port has to stay
faithful to is the scoring definition above, not any one copy of that
script: the block splitting and text normalization, the Hungarian
assignment over speaker blocks, and the word-level edit counts. A
regression test pins those by rescoring a recorded decode of the full test
set.

That recorded decode scores cpWER 28.31% here. The ESPnet2 recipe reports
27.95% for the same model decoded another way (the openai-whisper
``transcribe()`` path); this recipe decodes with ESPnet's own beam search
instead, see ``src/inference.py``. The two numbers belong to different
hypothesis sets, not to two different scorers.
"""

import importlib.util
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

if __package__:
    # Normal case: this module is part of the real egs3.ami.s2t.src.metrics
    # package, so the relative import resolves against it directly. Gating on
    # __package__ instead of wrapping this in try/except ImportError means a
    # genuine breakage in separator.py surfaces as its own ImportError here,
    # rather than being caught and silently rerouted into the fallback below.
    from ..separator import SPEAKER_CHANGE_SYMBOL
else:
    # A relative import has no parent package to resolve against when this
    # file is loaded standalone, for example via
    # ``importlib.util.spec_from_file_location`` in tests with a flat,
    # non-dotted module name (so __package__ is ``""``). Fall back to loading
    # separator.py by file path, freshly every time and with no sys.modules
    # caching: SPEAKER_CHANGE_SYMBOL must reflect whatever separator.py's
    # environment variable is set to at the moment this module is (re)loaded,
    # which is exactly what a test reloading this module after monkeypatching
    # the environment relies on.
    _spec = importlib.util.spec_from_file_location(
        f"{__name__}_ami_sot_separator_impl",
        Path(__file__).resolve().parent.parent / "separator.py",
    )
    _separator = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_separator)
    SPEAKER_CHANGE_SYMBOL = _separator.SPEAKER_CHANGE_SYMBOL

# The model separates speakers with a single BPE token, resolved once for the
# whole recipe as SPEAKER_CHANGE_SYMBOL (separator.py). This recipe's
# inference step (src/inference.py) normalizes it to "<sc>" before writing ref
# and hyp text, so "<sc>" is what this metric normally sees; the raw separator
# is still accepted for text that has not gone through that normalization.
# Both inference.py and this constant import the same symbol from
# separator.py, so it cannot drift out of sync between them.
_SEP_VARIANTS = ("<sc>", SPEAKER_CHANGE_SYMBOL)
_SEP = "▁SPKCHANGE▁"  # internal marker unlikely to occur in text
_SPECIAL_RE = re.compile(r"<\|[^|]*\|>")  # Whisper special tokens, e.g. <|1.20|>


def strip_special_tokens(text: str) -> str:
    """Remove Whisper special tokens (timestamps, <|endoftext|>, ...)."""
    return re.sub(r"\s+", " ", _SPECIAL_RE.sub(" ", text)).strip()


def split_speakers(text: str, cleaner) -> List[str]:
    """Split SOT text into per-speaker word lists.

    Both separator spellings are accepted, Whisper special tokens are stripped,
    the normalizer is applied, and empty blocks are dropped.
    """
    for v in _SEP_VARIANTS:
        text = text.replace(v, _SEP)
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


class CpWER(BaseMetric):
    """Concatenated minimum-permutation word error rate for SOT output.

    Each utterance group holds one text block per speaker, separated by
    ``<sc>``. The blocks arrive in an arbitrary order, so the score is taken
    over the assignment of hypothesis blocks to reference blocks that
    minimizes the total word distance.

    Args:
        ref_key: Alias of the reference SCP input.
        hyp_key: Alias of the hypothesis SCP input.
        clean_types: TextCleaner pipeline. The default matches the OWSM
            scoring convention the ESPnet2 recipe settled on.
    """

    def __init__(
        self,
        ref_key: str = "ref",
        hyp_key: str = "hyp",
        clean_types=("whisper_en",),
    ) -> None:
        self.ref_key = ref_key
        self.hyp_key = hyp_key
        self.cleaner = TextCleaner(list(clean_types) if clean_types else None)

    def __call__(self, data, test_name, output_dir):
        """Score one test set.

        Args:
            data: Alias to path mapping. Needs ``self.ref_key`` and
                ``self.hyp_key``, both SCP files with matching utterance ids
                in the same order.
            test_name: Test set name, used for the side-file directory.
            output_dir: Root the side files are written under.

        Returns:
            ``{"cpWER": <percentage rounded to two decimals>}``.

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
            ref_blocks = split_speakers(row[self.ref_key], self.cleaner)
            hyp_blocks = split_speakers(row[self.hyp_key], self.cleaner)
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
                "a perfect score instead of failing. The likely cause is a "
                "renamed dataset text key: src/inference.py's build_output "
                "reads data.get('text', \"\"), which silently returns an "
                f"empty string if the key it expects is missing. Check that "
                f"the {self.ref_key!r} input actually carries reference text."
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

        return {"cpWER": round(score, 2)}
