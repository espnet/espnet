#!/usr/bin/env python3
"""Add language labels to manifests from the existing Babel ASR preparer."""

import argparse
from pathlib import Path


def prepare(source, output, language):
    output.mkdir(parents=True, exist_ok=True)
    with (
        (source / "segments").open() as segments,
        (output / "segments").open("w") as out,
        (output / "utt2lang").open("w") as labels,
    ):
        for line in segments:
            utt, recording, start, end = line.split()
            out.write(f"{language}_{utt} {language}_{recording} {start} {end}\n")
            labels.write(f"{language}_{utt} {language}\n")
    with (source / "wav.scp").open() as wavs, (output / "wav.scp").open("w") as out:
        for line in wavs:
            recording, command = line.rstrip().split(maxsplit=1)
            out.write(f"{language}_{recording} {command}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("language")
    args = parser.parse_args()
    prepare(args.source, args.output, args.language)
