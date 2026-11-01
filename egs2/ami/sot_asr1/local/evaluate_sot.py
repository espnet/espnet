#!/usr/bin/env python3
"""Score SOT multi-talker ASR output: utterance-group cpWER + speaker counting.

Each utterance group is scored independently with its own optimal speaker
permutation. The permutation is the assignment (Hungarian algorithm) of
hypothesis speaker blocks to reference speaker blocks that minimizes the total
word error. Counts are then aggregated across all groups to give the
utterance-group cpWER. This is not session-level cpWER, where a single
permutation is found for a whole meeting.

Alongside cpWER this reports speaker-counting accuracy: how often the number of
hypothesized speaker blocks matches the number of reference speaker blocks.

Only dependencies already required by ESPnet are used: ``scipy`` and
``editdistance`` (both in ``setup.py``) for the assignment and word errors, and
``espnet2.text.cleaner.TextCleaner`` for normalization. ``--cleaner whisper_en``
maps to openai-whisper's ``EnglishTextNormalizer``, and openai-whisper is
already required by this recipe, so no external scoring toolkit is added.

Usage:
    python local/evaluate_sot.py \
        --hyp_text <decode_dir>/1best_recog/text \
        --ref_text data/test/text \
        --output_dir <out_dir> \
        --cleaner whisper_en
"""

import argparse
import json
import logging
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import editdistance
import numpy as np
from scipy.optimize import linear_sum_assignment

from espnet2.text.cleaner import TextCleaner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("evaluate_sot")

# The model separates speakers with a single BPE token that appears in text as
# "????" (the value of preprocessor_conf.speaker_change_symbol). local/decode.py
# rewrites it to "<sc>" in its output. Reference and hypothesis files may use
# either form, so both are normalized to a common marker before splitting.
_SEP_VARIANTS = ("<sc>", "????")
_SEP = "▁SPKCHANGE▁"  # internal marker unlikely to occur in text
_SPECIAL_RE = re.compile(r"<\|[^|]*\|>")  # Whisper special tokens, e.g. <|1.20|>


def strip_special_tokens(text: str) -> str:
    """Remove Whisper special tokens (timestamps, <|endoftext|>, ...)."""
    return re.sub(r"\s+", " ", _SPECIAL_RE.sub(" ", text)).strip()


def load_text(path: str) -> Dict[str, str]:
    """Load a Kaldi-format text file (``utt_id text``)."""
    out: Dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(None, 1)
            out[parts[0]] = parts[1] if len(parts) == 2 else ""
    return out


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


def main():
    parser = argparse.ArgumentParser(
        description="Score SOT output: utterance-group cpWER + speaker counting."
    )
    parser.add_argument("--hyp_text", required=True, help="Hypothesis text file.")
    parser.add_argument("--ref_text", required=True, help="Reference text file.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument(
        "--cleaner",
        default="whisper_en",
        help="espnet2 TextCleaner name (whisper_en removes fillers; "
        "whisper_basic keeps them; none disables normalization).",
    )
    args = parser.parse_args()

    cleaner = None if args.cleaner == "none" else TextCleaner([args.cleaner])
    logger.info(f"Text cleaner: {args.cleaner}")

    hyp_texts = load_text(args.hyp_text)
    ref_texts = load_text(args.ref_text)
    common = sorted(set(hyp_texts) & set(ref_texts))
    if not common:
        raise SystemExit("No common utterance ids between hypothesis and reference.")
    missing = set(ref_texts) - set(hyp_texts)
    if missing:
        logger.warning(f"{len(missing)} reference utts missing from hypothesis")
    logger.info(f"Scoring {len(common)} utterance groups")

    total = {"cor": 0, "sub": 0, "del": 0, "ins": 0, "ref_len": 0}
    by_nspk = defaultdict(
        lambda: {"cor": 0, "sub": 0, "del": 0, "ins": 0, "ref_len": 0}
    )
    spk_correct = 0
    spk_abs_err = 0
    spk_confusion = Counter()  # (ref_count, hyp_count) -> n
    spk_correct_by_nspk = Counter()
    per_utt = {}

    for uid in common:
        ref_blocks = split_speakers(ref_texts[uid], cleaner)
        hyp_blocks = split_speakers(hyp_texts[uid], cleaner)
        n_ref, n_hyp = len(ref_blocks), len(hyp_blocks)

        acc = group_cpwer(ref_blocks, hyp_blocks)
        for k in total:
            total[k] += acc[k]
            by_nspk[n_ref][k] += acc[k]

        spk_confusion[(n_ref, n_hyp)] += 1
        if n_ref == n_hyp:
            spk_correct += 1
            spk_correct_by_nspk[n_ref] += 1
        spk_abs_err += abs(n_ref - n_hyp)

        errs = acc["sub"] + acc["del"] + acc["ins"]
        per_utt[uid] = {
            "cpwer": pct(errs, acc["ref_len"]),
            "errors": errs,
            "ref_len": acc["ref_len"],
            "num_ref_speakers": n_ref,
            "num_hyp_speakers": n_hyp,
        }

    os.makedirs(args.output_dir, exist_ok=True)

    # --- cpWER (overall) ---
    errs = total["sub"] + total["del"] + total["ins"]
    cpwer = pct(errs, total["ref_len"])
    summary = {
        "cpwer": cpwer,
        "errors": errs,
        "ref_len": total["ref_len"],
        "substitutions": total["sub"],
        "deletions": total["del"],
        "insertions": total["ins"],
        "num_groups": len(common),
    }
    logger.info(
        f"cpWER: {cpwer:.2f}%  (S={total['sub']} D={total['del']} "
        f"I={total['ins']} / N={total['ref_len']})"
    )
    with open(os.path.join(args.output_dir, "cpwer.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # --- cpWER by reference speaker count ---
    cpwer_by_nspk = {}
    for n in sorted(by_nspk):
        d = by_nspk[n]
        e = d["sub"] + d["del"] + d["ins"]
        cpwer_by_nspk[n] = {
            "cpwer": pct(e, d["ref_len"]),
            "errors": e,
            "ref_len": d["ref_len"],
            "num_groups": spk_confusion_count(spk_confusion, n),
        }
        logger.info(f"  {n}-spk cpWER: {cpwer_by_nspk[n]['cpwer']:.2f}%")
    with open(os.path.join(args.output_dir, "cpwer_by_num_speakers.json"), "w") as f:
        json.dump(cpwer_by_nspk, f, indent=2)

    # --- speaker-counting accuracy ---
    n = len(common)
    spk_acc = pct(spk_correct, n)
    spk_summary = {
        "speaker_count_accuracy": spk_acc,
        "num_correct": spk_correct,
        "num_groups": n,
        "mean_abs_count_error": spk_abs_err / n if n else 0.0,
        "accuracy_by_num_ref_speakers": {
            str(k): {
                "accuracy": pct(
                    spk_correct_by_nspk[k], spk_confusion_count(spk_confusion, k)
                ),
                "num_groups": spk_confusion_count(spk_confusion, k),
            }
            for k in sorted(by_nspk)
        },
        "confusion_ref_by_hyp": {
            f"{r}->{h}": c for (r, h), c in sorted(spk_confusion.items())
        },
    }
    logger.info(
        f"Speaker-counting accuracy: {spk_acc:.2f}% "
        f"({spk_correct}/{n}); MAE={spk_summary['mean_abs_count_error']:.3f}"
    )
    with open(os.path.join(args.output_dir, "speaker_count.json"), "w") as f:
        json.dump(spk_summary, f, indent=2)

    with open(os.path.join(args.output_dir, "per_utt.json"), "w") as f:
        json.dump(per_utt, f, indent=2)

    logger.info(f"Results written to {args.output_dir}")


def spk_confusion_count(confusion: Counter, n_ref: int) -> int:
    return sum(c for (r, _), c in confusion.items() if r == n_ref)


if __name__ == "__main__":
    main()
