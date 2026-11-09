#!/usr/bin/env python3
"""Prepare one corpus (or one shard of it) as a VAD dump directory.

    python3 local/v2/prep.py --corpus fsd50k --root /path/to/corpora \
        --out dump/v2 [--shard 0 --nshards 4]

writes dump/v2/<corpus>/<split>[/shard<k>] for the splits the corpus has
(train, dev; see vad_dump.py). The corpora and how each is read are in
adapters.py.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapters import ADAPTERS  # noqa: E402
from vad_dump import VadDumpWriter  # noqa: E402


class LimitReached(Exception):
    """Raised once --limit recordings were added (a quick check of a reader)."""


class Writers:
    """One VadDumpWriter per split, created when the split gets its first recording."""

    def __init__(self, out, spec, corpus, fs, shard, nshards, pack_seconds, limit=None):
        self.out, self.spec, self.corpus, self.fs = out, spec, corpus, fs
        self.pack_seconds = pack_seconds
        self.limit = limit
        self.shard, self.nshards = shard, nshards
        self.writers = {}

    def __getitem__(self, split: str) -> VadDumpWriter:
        if split not in ("train", "dev"):
            raise ValueError(f"unknown split {split}")
        if split not in self.writers:
            d = self.out / self.corpus / split
            if self.nshards > 1:
                d = d / f"shard{self.shard}"
            self.writers[split] = VadDumpWriter(
                d,
                corpus=self.corpus,
                domain=self.spec.domain,
                label_source=self.spec.label_source,
                licence=self.spec.licence,
                fs=self.fs,
                pack_seconds=self.pack_seconds,
            )
        if self.limit is not None:
            added = sum(
                len(w.num_samples) + len(w._buffered) for w in self.writers.values()
            )
            if added >= self.limit:
                raise LimitReached()
        return self.writers[split]

    def close(self):
        for w in self.writers.values():
            w.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", required=True, choices=sorted(ADAPTERS))
    parser.add_argument("--root", required=True, type=Path, help="corpora root")
    parser.add_argument("--out", required=True, type=Path, help="dump root")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--fs", type=int, default=16000)
    parser.add_argument(
        "--pack_seconds",
        type=float,
        default=30.0,
        help="concatenate recordings shorter than this (0: never); the training "
        "chunks (10 s) need recordings at least that long",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="stop after about this many recordings (to check a reader)",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    spec = ADAPTERS[args.corpus]
    writers = Writers(
        args.out,
        spec,
        args.corpus,
        args.fs,
        args.shard,
        args.nshards,
        args.pack_seconds or None,
        args.limit,
    )
    try:
        spec.prepare(args.root / spec.directory, writers, args.shard, args.nshards)
    except LimitReached:
        logging.info("stopped at --limit %d", args.limit)
    writers.close()


if __name__ == "__main__":
    main()
