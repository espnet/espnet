#!/usr/bin/env python3
"""Generate ESPnet token list file from tiktoken encoding.

Usage:
    python local/generate_token_list.py \
        --output token_list.txt \
        --added_tokens_txt local/added_tokens.txt
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Generate token list from tiktoken")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--added_tokens_txt", type=str, default=None)
    parser.add_argument("--added_tokens", type=str, nargs="*", default=None)
    parser.add_argument(
        "--num_languages",
        type=int,
        default=99,
        help="Number of language tokens. "
        "99 for small/medium/large-v1/v2, "
        "100 for large-v3/large-v3-turbo (adds <|yue|>).",
    )
    args = parser.parse_args()

    from espnet2.train.sot_preprocessor import SOTWhisperPreprocessor

    SOTWhisperPreprocessor.generate_token_list(
        output_path=args.output,
        added_tokens_txt=args.added_tokens_txt,
        added_tokens=args.added_tokens,
        num_languages=args.num_languages,
    )


if __name__ == "__main__":
    main()
