#!/usr/bin/env python3
"""Convert manifests from the shared ML-SUPERB2 downloader to LID format."""

import argparse
from pathlib import Path


def prepare(source_dir, output_dir):
    for split in ("train", "dev", "dev_dialect"):
        source = source_dir / split
        wavs = dict(
            line.rstrip().split(maxsplit=1)
            for line in (source / "wav.scp").read_text().splitlines()
        )
        labels = {}
        for line in (source / "text").read_text().splitlines():
            utt, text = line.split(maxsplit=1)
            label = text.split("]", 1)[0].removeprefix("[")
            # GeoLID uses the macrolanguage label for Azerbaijani.
            labels[utt] = "aze" if label == "azj" else label
        if wavs.keys() != labels.keys():
            raise ValueError(f"Mismatched ML-SUPERB2 manifests: {split}")
        target = output_dir / f"{split}_ml_superb2_lang"
        target.mkdir(parents=True, exist_ok=True)
        with (
            (target / "wav.scp").open("w") as audio,
            (target / "utt2lang").open("w") as languages,
        ):
            for utt, lang in sorted(labels.items()):
                audio.write(f"{lang}_{utt} {wavs[utt]}\n")
                languages.write(f"{lang}_{utt} {lang}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", type=Path, default=Path("data"))
    parser.add_argument("--output_dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    prepare(args.source_dir, args.output_dir)
