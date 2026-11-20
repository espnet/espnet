#!/usr/bin/env python3
"""Split a key file into parts of about equal total length.

utils/split_scp.pl cuts the key file into contiguous blocks of equal line
counts. When the lengths of the entries differ a lot (e.g. hour-long meeting
recordings next to short clips), one block can hold most of the audio and its
job runs far longer than the others. This script assigns each line to the part
with the smallest total length so far, taking the longest entries first, and
writes every part in the order of the input file.

Usage:
    split_scp_by_length.py --lengths utt2num_samples wav.scp out.1.scp out.2.scp ...
"""

import argparse
import heapq
import logging
import statistics
from pathlib import Path
from typing import Dict, List


def read_lengths(path: Path) -> Dict[str, float]:
    """Key -> length from a file of "key length" lines."""
    lengths = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            fields = line.split()
            if len(fields) >= 2:
                lengths[fields[0]] = float(fields[1])
    return lengths


def assign(keys: List[str], lengths: Dict[str, float], num_parts: int) -> List[int]:
    """Part index of each key: the longest key goes to the lightest part first.

    Keys missing from `lengths` count as the median of the known lengths (1 if
    none is known). Ties go to the lower part index, so the result is
    deterministic. With at least `num_parts` keys, every part gets a key.
    """
    known = [lengths[k] for k in keys if k in lengths]
    fallback = statistics.median(known) if known else 1.0
    size = [lengths.get(k, fallback) for k in keys]
    order = sorted(range(len(keys)), key=lambda i: (-size[i], i))
    heap = [(0.0, part) for part in range(num_parts)]
    parts = [0] * len(keys)
    for i in order:
        total, part = heapq.heappop(heap)
        parts[i] = part
        heapq.heappush(heap, (total + size[i], part))
    return parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lengths",
        type=Path,
        required=True,
        help='file of "key length" lines, e.g. utt2num_samples',
    )
    parser.add_argument("key_file", type=Path, help="file to split (first field: key)")
    parser.add_argument("outputs", type=Path, nargs="+", help="output files")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    with args.key_file.open(encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    if len(lines) < len(args.outputs):
        raise SystemExit(
            f"{args.key_file} has {len(lines)} lines, fewer than "
            f"{len(args.outputs)} outputs"
        )
    keys = [line.split(maxsplit=1)[0] for line in lines]
    lengths = read_lengths(args.lengths)
    missing = sum(k not in lengths for k in keys)
    if missing:
        logging.warning(
            "%d of %d keys have no length in %s; they count as the median",
            missing,
            len(keys),
            args.lengths,
        )
    parts = assign(keys, lengths, len(args.outputs))

    files = [p.open("w", encoding="utf-8") for p in args.outputs]
    try:
        for line, part in zip(lines, parts):
            files[part].write(line)
    finally:
        for f in files:
            f.close()


if __name__ == "__main__":
    main()
