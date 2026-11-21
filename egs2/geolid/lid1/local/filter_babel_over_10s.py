#!/usr/bin/env python3
"""Filter materialized Babel waveforms, keeping language and duration metadata."""

import argparse
import shutil
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--babel_dir", type=Path, required=True)
    parser.add_argument("--babel_over_10s_dir", type=Path, required=True)
    args = parser.parse_args()
    source, target = args.babel_dir, args.babel_over_10s_dir
    if source.resolve() == target.resolve():
        parser.error("Input and output directories must differ.")
    keep = {
        key
        for key, samples in (
            line.split()
            for line in (source / "utt2num_samples").read_text().splitlines()
        )
        if int(samples) >= 10 * 16000
    }
    target.mkdir(parents=True, exist_ok=True)
    for name in ("wav.scp", "utt2lang", "utt2num_samples"):
        lines = [
            line
            for line in (source / name).read_text().splitlines()
            if line.split()[0] in keep
        ]
        (target / name).write_text("".join(line + "\n" for line in sorted(lines)))
    grouped = defaultdict(list)
    for line in (target / "utt2lang").read_text().splitlines():
        utt, lang = line.split()
        grouped[lang].append(utt)
    mapping = "".join(
        f"{lang} {' '.join(sorted(utts))}\n" for lang, utts in sorted(grouped.items())
    )
    for name in ("lang2utt", "category2utt"):
        (target / name).write_text(mapping)
    for name in ("audio_format", "feats_type"):
        shutil.copyfile(source / name, target / name)


if __name__ == "__main__":
    main()
