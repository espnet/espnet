#!/usr/bin/env python3
"""Heuristic filtering for caption data."""

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Set, Tuple

from nltk import ngrams


# =============================================================================
# Constants
# =============================================================================

FILTER_REASONS = [
    "finish_reason",
    "missing_fields",
    "non_latin",
    "char_repetition",
    "word_repetition",
    "ngram_repetition",
    "unique_word_ratio_low",
    "avg_word_length_low",
    "avg_word_length_high",
    "uppercase_ratio_high",
    "tokens_low",
    "tokens_high",
    "duration_low",
    "duration_high",
]

# Unicode ranges for non-Latin scripts (to be filtered out)
NON_LATIN_RANGES = [
    (0x0400, 0x04FF),    # Cyrillic
    (0x0500, 0x052F),    # Cyrillic Supplement
    (0x0600, 0x06FF),    # Arabic
    (0x0750, 0x077F),    # Arabic Supplement
    (0x0590, 0x05FF),    # Hebrew
    (0x0900, 0x097F),    # Devanagari
    (0x0980, 0x09FF),    # Bengali
    (0x0E00, 0x0E7F),    # Thai
    (0x1100, 0x11FF),    # Hangul Jamo (Korean)
    (0x3000, 0x303F),    # CJK Symbols and Punctuation
    (0x3040, 0x309F),    # Hiragana (Japanese)
    (0x30A0, 0x30FF),    # Katakana (Japanese)
    (0x3100, 0x312F),    # Bopomofo
    (0x3130, 0x318F),    # Hangul Compatibility Jamo
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs (Chinese/Japanese/Korean)
    (0xAC00, 0xD7AF),    # Hangul Syllables (Korean)
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
    (0xFF00, 0xFFEF),    # Halfwidth and Fullwidth Forms
]


# =============================================================================
# Filter Configuration
# =============================================================================

@dataclass
class FilterConfig:
    """Configuration for filtering thresholds."""

    max_char_repetition: int = 5
    max_word_repetition: int = 3
    max_ngram_repetition_ratio: float = 0.1
    ngram_n: int = 5
    min_unique_word_ratio: float = 0.3
    min_avg_word_length: float = 2.0
    max_avg_word_length: float = 15.0
    max_uppercase_ratio: float = 0.5
    min_tokens: int = 100
    max_tokens: int = 1024
    min_duration: float = 2.0
    max_duration: float = 30.0


# =============================================================================
# Helper Functions
# =============================================================================

def contains_non_latin(text: str) -> bool:
    """Check if text contains non-Latin script characters."""
    for char in text:
        code = ord(char)
        for start, end in NON_LATIN_RANGES:
            if start <= code <= end:
                return True
    return False


def get_max_char_repetition(text: str) -> int:
    """Get the maximum consecutive repeated character count."""
    if not text:
        return 0
    max_rep = 1
    for match in re.finditer(r"(.)\1+", text):
        rep_len = len(match.group())
        if rep_len > max_rep:
            max_rep = rep_len
    return max_rep


def get_max_word_repetition(words: List[str]) -> int:
    """Get the maximum consecutive repeated word count."""
    if not words:
        return 0
    max_rep = 1
    current_rep = 1
    for i in range(1, len(words)):
        if words[i].lower() == words[i - 1].lower():
            current_rep += 1
            max_rep = max(max_rep, current_rep)
        else:
            current_rep = 1
    return max_rep


def get_ngram_repetition_ratio(words: List[str], n: int) -> float:
    """Get the ratio of duplicate n-grams to total n-grams."""
    if len(words) < n:
        return 0.0
    ngram_list = list(ngrams(words, n))
    if not ngram_list:
        return 0.0
    total = len(ngram_list)
    unique = len(set(ngram_list))
    return 1.0 - (unique / total)


def get_unique_word_ratio(words: List[str]) -> float:
    """Get the ratio of unique words to total words."""
    if not words:
        return 0.0
    unique = len(set(w.lower() for w in words))
    return unique / len(words)


def get_avg_word_length(words: List[str]) -> float:
    """Get the average word length."""
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def get_uppercase_ratio(text: str) -> float:
    """Get the ratio of uppercase letters to total letters."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    uppercase = sum(1 for c in letters if c.isupper())
    return uppercase / len(letters)


# =============================================================================
# Filter Application
# =============================================================================

def apply_filters(record: dict, config: FilterConfig) -> Tuple[bool, str]:
    """
    Apply all filtering strategies to a record.

    Args:
        record: A single record from the jsonl file.
        config: Filter configuration with thresholds.

    Returns:
        Tuple of (should_delete, reason). reason is empty if not deleted.
    """
    # 1. Check finish_reason
    if record.get("finish_reason") != "stop":
        return True, "finish_reason"

    # 2. Check missing fields
    utt_id = record.get("utt_id")
    duration = record.get("duration")
    usage = record.get("usage", {})
    completion_tokens = usage.get("completion_tokens")
    caption = record.get("caption", "")

    if utt_id is None or duration is None or completion_tokens is None:
        return True, "missing_fields"

    # 3. Check non-Latin characters
    if contains_non_latin(caption):
        return True, "non_latin"

    # Tokenize caption for word-based checks
    words = caption.split()

    # Early return for empty caption (avoid misleading filter reasons)
    if not words:
        return True, "missing_fields"

    # 4. Check character repetition
    if get_max_char_repetition(caption) > config.max_char_repetition:
        return True, "char_repetition"

    # 5. Check word repetition
    if get_max_word_repetition(words) > config.max_word_repetition:
        return True, "word_repetition"

    # 6. Check n-gram repetition ratio
    ngram_ratio = get_ngram_repetition_ratio(words, config.ngram_n)
    if ngram_ratio > config.max_ngram_repetition_ratio:
        return True, "ngram_repetition"

    # 7. Check unique word ratio (lexical diversity)
    if get_unique_word_ratio(words) < config.min_unique_word_ratio:
        return True, "unique_word_ratio_low"

    # 8. Check average word length (too low)
    avg_len = get_avg_word_length(words)
    if avg_len < config.min_avg_word_length:
        return True, "avg_word_length_low"

    # 9. Check average word length (too high)
    if avg_len > config.max_avg_word_length:
        return True, "avg_word_length_high"

    # 10. Check uppercase ratio
    if get_uppercase_ratio(caption) > config.max_uppercase_ratio:
        return True, "uppercase_ratio_high"

    # 11. Check tokens (too low)
    if completion_tokens < config.min_tokens:
        return True, "tokens_low"

    # 12. Check tokens (too high)
    if completion_tokens > config.max_tokens:
        return True, "tokens_high"

    # 13. Check duration (too low)
    if duration < config.min_duration:
        return True, "duration_low"

    # 14. Check duration (too high)
    if duration > config.max_duration:
        return True, "duration_high"

    return False, ""


# =============================================================================
# Worker Function
# =============================================================================

def process_file(
    filepath: str,
    config: FilterConfig,
    max_demos_per_reason: int = 3,
) -> Tuple[Set[str], Dict[str, int], Dict[str, List[dict]]]:
    """
    Process a single jsonl file and return IDs to delete.

    Args:
        filepath: Path to a captions_rank*.jsonl file.
        config: Filter configuration with thresholds.
        max_demos_per_reason: Max demo examples to collect per reason.

    Returns:
        Tuple of (delete_ids, stats, demos).
    """
    delete_ids = set()
    stats = {"total": 0, "deleted": 0}
    demos = {}
    for reason in FILTER_REASONS:
        stats[reason] = 0
        demos[reason] = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            stats["total"] += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["missing_fields"] += 1
                stats["deleted"] += 1
                continue

            record_id = record.get("utt_id")
            should_delete, reason = apply_filters(record, config)

            if should_delete:
                stats["deleted"] += 1
                stats[reason] += 1
                if record_id:
                    delete_ids.add(record_id)
                # Collect demo examples
                if len(demos[reason]) < max_demos_per_reason:
                    demos[reason].append({
                        "reason": reason,
                        "utt_id": record_id,
                        "caption": record.get("caption", ""),
                        "duration": record.get("duration"),
                        "completion_tokens": record.get("usage", {}).get(
                            "completion_tokens"
                        ),
                    })

    survived = stats["total"] - stats["deleted"]
    print(f"Processed {filepath}: {survived}/{stats['total']} survived")

    return delete_ids, stats, demos


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Heuristic filtering for caption data."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing captions_rank*.jsonl files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save delete_ids.jsonl.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers.",
    )
    # Filter threshold arguments
    parser.add_argument(
        "--max_char_repetition",
        type=int,
        default=5,
        help="Max consecutive repeated characters (default: 5).",
    )
    parser.add_argument(
        "--max_word_repetition",
        type=int,
        default=3,
        help="Max consecutive repeated words (default: 3).",
    )
    parser.add_argument(
        "--max_ngram_repetition_ratio",
        type=float,
        default=0.1,
        help="Max n-gram repetition ratio (default: 0.1).",
    )
    parser.add_argument(
        "--ngram_n",
        type=int,
        default=5,
        help="N-gram size for repetition check (default: 5).",
    )
    parser.add_argument(
        "--min_unique_word_ratio",
        type=float,
        default=0.3,
        help="Min unique word ratio (default: 0.3).",
    )
    parser.add_argument(
        "--min_avg_word_length",
        type=float,
        default=2.0,
        help="Min average word length (default: 2.0).",
    )
    parser.add_argument(
        "--max_avg_word_length",
        type=float,
        default=15.0,
        help="Max average word length (default: 15.0).",
    )
    parser.add_argument(
        "--max_uppercase_ratio",
        type=float,
        default=0.5,
        help="Max uppercase letter ratio (default: 0.5).",
    )
    parser.add_argument(
        "--min_tokens",
        type=int,
        default=200,
        help="Min caption tokens (default: 200).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=800,
        help="Max caption tokens (default: 800).",
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=2.0,
        help="Min audio duration in seconds (default: 2.0).",
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=30.0,
        help="Max audio duration in seconds (default: 30.0).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: only process 5%% of files.",
    )
    args = parser.parse_args()

    # Check for done flag (skip if exists, unless debug mode)
    done_flag = os.path.join(args.output_dir, ".done")
    if os.path.exists(done_flag) and not args.debug:
        print(f"Skipping: already done ({done_flag} exists)")
        return

    # Build filter config
    config = FilterConfig(
        max_char_repetition=args.max_char_repetition,
        max_word_repetition=args.max_word_repetition,
        max_ngram_repetition_ratio=args.max_ngram_repetition_ratio,
        ngram_n=args.ngram_n,
        min_unique_word_ratio=args.min_unique_word_ratio,
        min_avg_word_length=args.min_avg_word_length,
        max_avg_word_length=args.max_avg_word_length,
        max_uppercase_ratio=args.max_uppercase_ratio,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )

    # Find all captions_rank*.jsonl files
    pattern = os.path.join(args.input_dir, "captions_rank*.jsonl")
    filepaths = sorted(glob.glob(pattern))

    if not filepaths:
        print(f"No files found matching {pattern}")
        return

    # Debug mode: only process 5% of files
    if args.debug:
        num_files = max(1, len(filepaths) // 20)
        filepaths = filepaths[:num_files]
        print(f"[DEBUG] Processing {len(filepaths)} files (5%)")
    else:
        print(f"Found {len(filepaths)} files to process")

    # Process files in parallel
    worker_fn = partial(process_file, config=config)
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(worker_fn, filepaths)

    # Merge results
    all_delete_ids = set()
    total_stats = {"total": 0, "deleted": 0}
    all_demos = {}
    for reason in FILTER_REASONS:
        total_stats[reason] = 0
        all_demos[reason] = []

    for delete_ids, stats, demos in results:
        all_delete_ids.update(delete_ids)
        for key in total_stats:
            total_stats[key] += stats[key]
        # Merge demos (keep up to 3 per reason)
        for reason in FILTER_REASONS:
            if len(all_demos[reason]) < 3:
                remaining = 3 - len(all_demos[reason])
                all_demos[reason].extend(demos[reason][:remaining])

    # Write delete IDs
    output_path = os.path.join(args.output_dir, "delete_ids.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for record_id in sorted(all_delete_ids):
            f.write(json.dumps({"id": record_id}) + "\n")

    # Compute summary
    survived = total_stats["total"] - total_stats["deleted"]
    survival_rate = 100 * survived / max(1, total_stats["total"])

    print("\n" + "=" * 60)
    print("Heuristic Filtering Summary")
    print("=" * 60)
    print(f"Total samples processed:    {total_stats['total']:,}")
    print(f"Samples survived:           {survived:,}")
    print(f"Samples deleted:            {total_stats['deleted']:,}")
    print(f"Survival rate:              {survival_rate:.2f}%")
    print("-" * 60)
    print("Deleted by reason:")
    print(f"  - finish_reason:          {total_stats['finish_reason']:,}")
    print(f"  - missing_fields:         {total_stats['missing_fields']:,}")
    print(f"  - non_latin:              {total_stats['non_latin']:,}")
    print(f"  - char_repetition:        {total_stats['char_repetition']:,}")
    print(f"  - word_repetition:        {total_stats['word_repetition']:,}")
    print(f"  - ngram_repetition:       {total_stats['ngram_repetition']:,}")
    print(f"  - unique_word_ratio_low:  {total_stats['unique_word_ratio_low']:,}")
    print(f"  - avg_word_length_low:    {total_stats['avg_word_length_low']:,}")
    print(f"  - avg_word_length_high:   {total_stats['avg_word_length_high']:,}")
    print(f"  - uppercase_ratio_high:   {total_stats['uppercase_ratio_high']:,}")
    print(f"  - tokens_low:             {total_stats['tokens_low']:,}")
    print(f"  - tokens_high:            {total_stats['tokens_high']:,}")
    print(f"  - duration_low:           {total_stats['duration_low']:,}")
    print(f"  - duration_high:          {total_stats['duration_high']:,}")
    print("=" * 60)
    print(f"Output saved to: {output_path}")

    # Save stats to JSON file
    stats_path = os.path.join(args.output_dir, "stats.json")
    stats_output = {
        "total": total_stats["total"],
        "survived": survived,
        "deleted": total_stats["deleted"],
        "survival_rate": round(survival_rate, 2),
        "deleted_by_reason": {
            reason: total_stats[reason] for reason in FILTER_REASONS
        },
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_output, f, indent=2)
    print(f"Stats saved to: {stats_path}")

    # Save demo examples to JSONL file
    demo_path = os.path.join(args.output_dir, "demo.jsonl")
    with open(demo_path, "w", encoding="utf-8") as f:
        for reason in FILTER_REASONS:
            for demo in all_demos[reason]:
                f.write(json.dumps(demo) + "\n")
    print(f"Demo examples saved to: {demo_path}")

    # Mark as done (only if not debug mode)
    if not args.debug:
        with open(done_flag, "w") as f:
            f.write("")
        print(f"Done flag created: {done_flag}")


if __name__ == "__main__":
    main()
