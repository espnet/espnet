#!/usr/bin/env python3

from __future__ import annotations

import argparse
import collections
from pathlib import Path


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def read_text(data_dir: Path):
    text_path = data_dir / "text"
    if not text_path.is_file():
        raise FileNotFoundError(f"Missing text file: {text_path}")

    entries = collections.defaultdict(list)
    with text_path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            fields = line.split(maxsplit=1)
            if len(fields) != 2:
                raise RuntimeError(f"{text_path}:{line_no}: expected '<utt> <text>'")
            utt_id, text = fields
            entries[normalize_text(text)].append((utt_id, text))
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("splits", nargs="+")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    split_text = {split: read_text(data_dir / split) for split in args.splits}

    overlaps = []
    for i, first in enumerate(args.splits):
        for second in args.splits[i + 1 :]:
            shared = sorted(set(split_text[first]) & set(split_text[second]))
            for text in shared:
                overlaps.append((first, second, text))

    if overlaps:
        print("Found transcript overlap:")
        for first, second, text in overlaps[:20]:
            first_utts = ", ".join(utt for utt, _ in split_text[first][text][:3])
            second_utts = ", ".join(utt for utt, _ in split_text[second][text][:3])
            print(f"{first} {second}: {text}")
            print(f"{first}: {first_utts}")
            print(f"{second}: {second_utts}")
        if len(overlaps) > 20:
            print(f"... {len(overlaps) - 20} more overlaps")
        return 1

    print("No transcript overlap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
