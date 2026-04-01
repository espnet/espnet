#!/usr/bin/env python3
"""Evaluate SOT multi-talker ASR output using cpWER.

Reads hypothesis and reference text files, splits on <sc> to get
per-speaker texts, and computes cpWER using meeteval.

Usage:
    python local/evaluate_sot.py \
        --hyp_text decode_dir/1best_recog/text \
        --ref_text data/dev/text \
        --output_dir results/dev \
        --speaker_change_token "<sc>"
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def strip_whisper_special_tokens(text: str) -> str:
    """Remove Whisper special tokens (timestamps, etc.)."""
    # Remove <|...|> tokens
    text = re.sub(r"<\|[^|]*\|>", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def truncate_at_repeating_ngram(
    text: str,
    ngram_length: int = 10,
    min_n: int = 1,
    max_n: int = None,
    min_word_threshold: int = 30,
    unigram_min_repeat: int = 10,
    repeat_threshold: int = 10,
) -> str:
    """Truncate text at the first occurrence of a repeating n-gram.

    Ported from TS-ASR-Whisper/src/data/postprocess.py.
    """
    if max_n is None:
        max_n = ngram_length

    words = text.split()
    if len(words) < min_word_threshold:
        return text

    earliest_truncation_idx = len(words)

    # Handle unigrams with consecutive repetition
    if min_n == 1:
        for i in range(len(words) - unigram_min_repeat + 1):
            current_word = words[i].lower()
            consecutive_count = 1
            for j in range(i + 1, len(words)):
                if words[j].lower() == current_word:
                    consecutive_count += 1
                else:
                    break
            if consecutive_count >= unigram_min_repeat:
                earliest_truncation_idx = min(earliest_truncation_idx, i + 1)
                break

    # Count all n-grams
    ngram_counts = defaultdict(int)
    for n in range(max(2, min_n), max_n + 1):
        for i in range(len(words) - n + 1):
            ngram_words = words[i : i + n]
            if all(w.lower() == ngram_words[0].lower() for w in ngram_words):
                continue
            ngram_counts[" ".join(ngram_words)] += 1

    # Find earliest occurrence of any repeated n-gram above threshold
    lengths_to_check = [ngram_length] + [
        n for n in range(min_n, max_n + 1) if n != ngram_length and n > 1
    ]
    for n in lengths_to_check:
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i : i + n])
            if ngram_counts[ngram] > repeat_threshold:
                earliest_truncation_idx = min(earliest_truncation_idx, i + n)

    if earliest_truncation_idx < len(words):
        return " ".join(words[:earliest_truncation_idx])
    return text


def load_text_file(path: str) -> Dict[str, str]:
    """Load Kaldi-format text file (utt_id text)."""
    entries = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                utt_id, text = parts
            else:
                utt_id = parts[0]
                text = ""
            entries[utt_id] = text
    return entries


def split_sot_text(text: str, speaker_change_token: str, text_norm=None) -> List[str]:
    """Split SOT text at speaker change tokens to get per-speaker texts.

    Also strips Whisper special tokens from each speaker's text.
    """
    parts = text.split(speaker_change_token)
    speakers = []
    for part in parts:
        cleaned = strip_whisper_special_tokens(part)
        if cleaned:
            cleaned = truncate_at_repeating_ngram(cleaned)
            if text_norm is not None:
                cleaned = text_norm(cleaned)
            if cleaned:
                speakers.append(cleaned)
    return speakers


def compute_cpwer(
    refs: Dict[str, List[str]],
    hyps: Dict[str, List[str]],
) -> dict:
    """Compute cpWER using meeteval.

    Args:
        refs: dict of utt_id -> list of reference speaker texts
        hyps: dict of utt_id -> list of hypothesis speaker texts

    Returns:
        dict with cpWER metrics
    """
    import meeteval

    cp_wer = meeteval.wer.wer.cp_word_error_rate_multifile(
        reference=refs, hypothesis=hyps
    )
    combined = meeteval.wer.combine_error_rates(cp_wer)

    return {
        "per_utt": cp_wer,
        "combined": combined,
        "cpwer": combined.error_rate,
        "errors": combined.errors,
        "length": combined.length,
        "insertions": combined.insertions,
        "deletions": combined.deletions,
        "substitutions": combined.substitutions,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SOT output using cpWER.",
    )
    parser.add_argument(
        "--hyp_text",
        type=str,
        required=True,
        help="Path to hypothesis text file (Kaldi format: utt_id text)",
    )
    parser.add_argument(
        "--ref_text",
        type=str,
        required=True,
        help="Path to reference text file (Kaldi format: utt_id text)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--speaker_change_token",
        type=str,
        default="????",
        help="Token used to separate speakers in SOT output",
    )
    parser.add_argument(
        "--text_norm",
        type=str,
        default="whisper_nsf_keep_fillers",
        help="Text normalizer: whisper_nsf_keep_fillers (default) or none",
    )
    args = parser.parse_args()

    # Set up text normalizer
    text_norm = None
    if args.text_norm and args.text_norm != "none":
        sys.path.insert(0, "/work/nvme/bbjs/chuang14/mtasr/TS-ASR-Whisper/src")
        from txt_norm import get_text_norm

        text_norm = get_text_norm(args.text_norm)
        logger.info(f"Text normalizer: {args.text_norm}")

    # Load files
    hyp_texts = load_text_file(args.hyp_text)
    ref_texts = load_text_file(args.ref_text)

    logger.info(f"Loaded {len(hyp_texts)} hypotheses, {len(ref_texts)} references")

    # Find common utterances
    common_utts = set(hyp_texts.keys()) & set(ref_texts.keys())
    if len(common_utts) == 0:
        logger.error("No common utterance IDs between hypothesis and reference!")
        return

    missing_hyp = set(ref_texts.keys()) - set(hyp_texts.keys())
    missing_ref = set(hyp_texts.keys()) - set(ref_texts.keys())
    if missing_hyp:
        logger.warning(
            f"{len(missing_hyp)} reference utterances missing from hypothesis"
        )
    if missing_ref:
        logger.warning(
            f"{len(missing_ref)} hypothesis utterances missing from reference"
        )

    # Split SOT texts into per-speaker parts
    refs = {}
    hyps = {}
    num_ref_speakers = {}
    for utt_id in sorted(common_utts):
        ref_speakers = split_sot_text(
            ref_texts[utt_id], args.speaker_change_token, text_norm
        )
        hyp_speakers = split_sot_text(
            hyp_texts[utt_id], args.speaker_change_token, text_norm
        )

        # meeteval expects non-empty lists
        if not ref_speakers:
            ref_speakers = [""]
        if not hyp_speakers:
            hyp_speakers = [""]

        # Cap hypothesis speakers to avoid meeteval crash on hallucinated
        # speaker changes (keep first N blocks where N = 2 * num_ref_speakers)
        max_hyp_spk = max(len(ref_speakers) * 2, 5)
        if len(hyp_speakers) > max_hyp_spk:
            logger.warning(
                f"{utt_id}: capping hyp speakers from "
                f"{len(hyp_speakers)} to {max_hyp_spk}"
            )
            hyp_speakers = hyp_speakers[:max_hyp_spk]

        refs[utt_id] = ref_speakers
        hyps[utt_id] = hyp_speakers
        num_ref_speakers[utt_id] = len(ref_speakers)

    logger.info(f"Evaluating {len(refs)} utterances")

    # Compute cpWER
    results = compute_cpwer(refs, hyps)

    # Output
    os.makedirs(args.output_dir, exist_ok=True)

    # Overall cpWER
    logger.info(f"cpWER: {results['cpwer']:.4f}")
    logger.info(f"  errors={results['errors']}, length={results['length']}")
    logger.info(
        f"  ins={results['insertions']}, del={results['deletions']}, "
        f"sub={results['substitutions']}"
    )

    with open(os.path.join(args.output_dir, "cpwer.json"), "w") as f:
        json.dump(
            {
                "cpwer": results["cpwer"],
                "errors": results["errors"],
                "length": results["length"],
                "insertions": results["insertions"],
                "deletions": results["deletions"],
                "substitutions": results["substitutions"],
            },
            f,
            indent=2,
        )

    # Per-speaker-count breakdown
    spk_groups = defaultdict(list)
    for utt_id, n in num_ref_speakers.items():
        spk_groups[n].append(utt_id)

    cpwer_by_nspk = {}
    import meeteval

    for n, utt_ids in sorted(spk_groups.items()):
        group_refs = {uid: refs[uid] for uid in utt_ids}
        group_hyps = {uid: hyps[uid] for uid in utt_ids}
        group_cp = meeteval.wer.wer.cp_word_error_rate_multifile(
            reference=group_refs, hypothesis=group_hyps
        )
        group_combined = meeteval.wer.combine_error_rates(group_cp)
        cpwer_by_nspk[n] = {
            "cpwer": group_combined.error_rate,
            "errors": group_combined.errors,
            "length": group_combined.length,
            "num_sessions": len(utt_ids),
        }
        logger.info(
            f"  {n} spk(s): cpWER={group_combined.error_rate:.4f} "
            f"(n={len(utt_ids)})"
        )

    with open(os.path.join(args.output_dir, "cpwer_by_num_speakers.json"), "w") as f:
        json.dump(cpwer_by_nspk, f, indent=2)

    # Save per-utterance results
    per_utt_results = {}
    for utt_id in sorted(refs.keys()):
        result = results["per_utt"][utt_id]
        per_utt_results[utt_id] = {
            "cpwer": result.error_rate,
            "errors": result.errors,
            "length": result.length,
            "num_ref_speakers": num_ref_speakers[utt_id],
            "num_hyp_speakers": len(hyps[utt_id]),
        }

    with open(os.path.join(args.output_dir, "per_utt_cpwer.json"), "w") as f:
        json.dump(per_utt_results, f, indent=2)

    # Save hyp/ref for inspection
    with open(os.path.join(args.output_dir, "hyp.json"), "w") as f:
        json.dump(hyps, f, indent=2)
    with open(os.path.join(args.output_dir, "ref.json"), "w") as f:
        json.dump(refs, f, indent=2)

    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
