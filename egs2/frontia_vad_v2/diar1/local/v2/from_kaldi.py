#!/usr/bin/env python3
"""Turn a Kaldi VAD data directory (frontia_vad_v1 local/prep_*.py) into a v2 dump.

    python3 local/v2/from_kaldi.py data/ami_train dump/v2/ami/train \
        --corpus ami --domain meeting --licence "CC BY 4.0" [--shard 0 --nshards 8]

The v1 preparation writes wav.scp (a path, or a command ending in "|" such as
sph2pipe or sox remix), segments and utt2spk with one pseudo speaker per
recording. Each recording is read once and written to the Omni-IO archive of
the dump; its speech is the union of its segments (label source "H").
"""

import argparse
import io
import logging
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import soundfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vad_dump import VadDumpWriter  # noqa: E402


def read_entry(entry: str):
    """(samples x channels, rate) of a wav.scp entry: a path or a command with "|"."""
    if entry.rstrip().endswith("|"):
        out = subprocess.run(
            entry.rstrip()[:-1], shell=True, check=True, capture_output=True
        ).stdout
        return soundfile.read(io.BytesIO(out), dtype="float32", always_2d=True)
    return soundfile.read(entry, dtype="float32", always_2d=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--licence", required=True)
    parser.add_argument("--mic", default="")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    wavs = dict(
        line.rstrip("\n").split(maxsplit=1)
        for line in (args.data_dir / "wav.scp").open()
        if line.strip()
    )
    speech = defaultdict(list)
    for line in (args.data_dir / "segments").open():
        _, rec, start, end = line.split()
        speech[rec].append((float(start), float(end)))
    out = args.out / f"shard{args.shard}" if args.nshards > 1 else args.out
    with VadDumpWriter(out, args.corpus, args.domain, "H", args.licence) as w:
        for i, rec in enumerate(sorted(wavs)):
            if i % args.nshards != args.shard:
                continue
            try:
                x, rate = read_entry(wavs[rec])
            except Exception as e:  # noqa: BLE001 - unreadable or truncated file
                w.skip(wavs[rec], f"{type(e).__name__}: {e}")
                continue
            w.add(
                rec,
                x[:, :1],
                rate,
                speech=speech.get(rec, []),
                mic=args.mic,
                source=wavs[rec],
            )


if __name__ == "__main__":
    main()
