#!/usr/bin/env python3
import sys
import argparse
from collections import defaultdict
from pathlib import Path

from espnet.utils.cli_utils import get_commandline_args
from typing import Optional
from tqdm import tqdm


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Insert silence token",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--split_dir", type=str, required=True)
    parser.add_argument("--num_splits", type=int, required=True)
    parser.add_argument("--output_dir", "-o", required=True, help="Output dir")
    parser.add_argument(
        "--add_symbol", type=str, help="extra symbol add to vocabulary", action="append"
    )
    return parser


def combine(
    split_dir: str,
    num_splits: int,
    output_dir: str,
    add_symbol: Optional[str],
):
    extra_symbol_set = set()
    if add_symbol:
        extra_symbol_set = set(add_symbol)

    output_text = Path(output_dir, "phones.txt")
    output_text.parent.mkdir(parents=True, exist_ok=True)
    ot_f = output_text.open("w", encoding="utf-8")

    output_vocab = Path(output_dir, "tokens.txt")
    output_vocab.parent.mkdir(parents=True, exist_ok=True)
    ov_f = output_vocab.open("w", encoding="utf-8")

    vocab_dict = defaultdict(int)
    for i in tqdm(range(num_splits)):
        index = i + 1
        text_f = Path(split_dir, str(index), "phones.txt").open("r", encoding="utf-8")
        for line in text_f:
            line = line.rstrip()
            if len(line) > 0:
                for token in line.split():
                    vocab_dict[token] += 1
            ot_f.write(f"{line}\n")
    ot_f.close()

    for token in extra_symbol_set:
        if token not in vocab_dict:
            ov_f.write(f"{token}\n")
    for token, counts in vocab_dict.items():
        ov_f.write(f"{token}\n")
    ov_f.close()


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)

    combine(**kwargs)


if __name__ == "__main__":
    main()
