#!/usr/bin/env python3
"""
Filter text utterances based on repetitive patterns and long digit sequences.

This script detects and filters out utterances containing:
1. Excessive character repetition (e.g., "Ooooooooo...")
2. Excessive word/phrase repetition (e.g., "uh, uh, uh...")
3. High ratio of repetitive content
4. Long sequences of digits (e.g., "1234567890")

Usage:
    python filter_text.py "text to check" [options]
    echo "text to check" | python filter_text.py [options]

Exit codes:
    0: Text is acceptable (should be kept)
    1: Text should be filtered out
    2: Error in processing
"""

import argparse
import re
import sys
from collections import Counter


def detect_char_repetition(text, max_repeat=10):
    """
    Detect excessive character repetition.

    Args:
        text: Input text
        max_repeat: Maximum allowed consecutive character repetitions

    Returns:
        True if excessive repetition detected, False otherwise
    """
    # Pattern: same character repeated more than max_repeat times
    pattern = r"(.)\1{" + str(max_repeat) + r",}"
    matches = re.findall(pattern, text, re.IGNORECASE)

    if matches:
        return True
    return False


def detect_word_repetition(text, max_repeat=5, max_word_len=3):
    """
    Detect excessive word/phrase repetition.

    Args:
        text: Input text
        max_repeat: Maximum allowed consecutive word repetitions
        max_word_len: Maximum word length to consider for repetition detection

    Returns:
        True if excessive repetition detected, False otherwise
    """
    # Split by common delimiters
    words = re.split(r"[\s,;.!?]+", text.lower())
    words = [w for w in words if w]  # Remove empty strings

    if len(words) < max_repeat:
        return False

    # Check for consecutive repetitions of short words/phrases
    consecutive_count = 1
    prev_word = None

    for word in words:
        # Only check short words (likely fillers like "uh", "ah", etc.)
        if len(word) <= max_word_len:
            if word == prev_word:
                consecutive_count += 1
                if consecutive_count > max_repeat:
                    return True
            else:
                consecutive_count = 1
                prev_word = word
        else:
            consecutive_count = 1
            prev_word = None

    return False


def detect_repetition_ratio(text, max_ratio=0.5):
    """
    Detect if the text has too high a ratio of repetitive content.

    Args:
        text: Input text
        max_ratio: Maximum allowed ratio of repetitive characters

    Returns:
        True if repetition ratio exceeds threshold, False otherwise
    """
    if len(text) == 0:
        return True

    # Count character frequencies (case-insensitive, excluding spaces)
    text_clean = re.sub(r"\s+", "", text.lower())

    if len(text_clean) == 0:
        return True

    char_counts = Counter(text_clean)

    # Calculate ratio of most common character
    if char_counts:
        most_common_count = char_counts.most_common(1)[0][1]
        ratio = most_common_count / len(text_clean)

        if ratio > max_ratio:
            return True

    return False


def detect_long_digits(text, threshold=15):
    """
    Detect lines containing long digit runs.
    Matches raw digits or digits separated by common punctuation (commas, dots, etc).
    """
    if threshold <= 0:
        return False

    patt = re.compile(r"\d{" + str(threshold) + r",}")

    # Check raw text
    if patt.search(text):
        return True

    # Check "clean" text (handling formatted numbers like 123,456,789...)
    clean = text.replace(",", "").replace("|", "").replace("'", "").replace(".", "")
    if patt.search(clean):
        return True

    return False


def should_filter(
    text,
    max_char_repeat=10,
    max_word_repeat=5,
    max_repeat_ratio=0.5,
    max_long_digits=15,
    verbose=False,
):
    """
    Determine if text should be filtered out.

    Args:
        text: Input text to check
        max_char_repeat: Maximum consecutive character repetitions
        max_word_repeat: Maximum consecutive word repetitions
        max_repeat_ratio: Maximum ratio of repetitive characters
        max_long_digits: Maximum length of digit sequences
        verbose: Print reason for filtering

    Returns:
        True if text should be filtered, False if it should be kept
    """
    # Empty text should be filtered
    if not text or not text.strip():
        if verbose:
            print("FILTER: Empty text", file=sys.stderr)
        return True

    # Check character repetition
    if detect_char_repetition(text, max_char_repeat):
        if verbose:
            print(
                f"FILTER: Excessive character repetition (>{max_char_repeat})",
                file=sys.stderr,
            )
        return True

    # Check word repetition
    if detect_word_repetition(text, max_word_repeat):
        if verbose:
            print(
                f"FILTER: Excessive word repetition (>{max_word_repeat})",
                file=sys.stderr,
            )
        return True

    # Check repetition ratio
    if detect_repetition_ratio(text, max_repeat_ratio):
        if verbose:
            print(
                f"FILTER: High repetition ratio (>{max_repeat_ratio})", file=sys.stderr
            )
        return True

    # Check long digit sequences
    if detect_long_digits(text, max_long_digits):
        if verbose:
            print(
                f"FILTER: Long digit sequence (>{max_long_digits} digits)",
                file=sys.stderr,
            )
        return True

    if verbose:
        print("KEEP: Text passes all filters", file=sys.stderr)

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Filter text based on repetitive patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Check a specific text
    python filter_text.py "This is normal text"

    # Check text from stdin
    echo "Ooooooooooooooooo" | python filter_text.py

    # Use custom thresholds
    python filter_text.py "uh uh uh" --max_word_repeat 2

Exit codes:
    0: Text should be kept
    1: Text should be filtered out
    2: Error
        """,
    )

    parser.add_argument(
        "text", nargs="?", help="Text to check (if not provided, reads from stdin)"
    )
    parser.add_argument(
        "--max_char_repeat",
        type=int,
        default=10,
        help="Maximum consecutive character repetitions (default: 10)",
    )
    parser.add_argument(
        "--max_word_repeat",
        type=int,
        default=5,
        help="Maximum consecutive word repetitions (default: 5)",
    )
    parser.add_argument(
        "--max_repeat_ratio",
        type=float,
        default=0.5,
        help="Maximum ratio of repetitive characters (default: 0.5)",
    )
    parser.add_argument(
        "--max_long_digits",
        type=int,
        default=15,
        help="Maximum length of digit sequences (default: 15)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print filtering reason to stderr"
    )

    args = parser.parse_args()

    # Get text from argument or stdin
    if args.text:
        text = args.text
    else:
        text = sys.stdin.read().strip()

    # Check if text should be filtered
    try:
        should_be_filtered = should_filter(
            text,
            max_char_repeat=args.max_char_repeat,
            max_word_repeat=args.max_word_repeat,
            max_repeat_ratio=args.max_repeat_ratio,
            max_long_digits=args.max_long_digits,
            verbose=args.verbose,
        )

        # Exit with appropriate code
        sys.exit(1 if should_be_filtered else 0)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
