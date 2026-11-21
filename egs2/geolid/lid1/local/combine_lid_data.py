#!/usr/bin/env python3
"""Combine materialized LID datasets without changing the source directories."""

import argparse
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()
    merged = {
        name: {} for name in ("wav.scp", "utt2lang", "utt2num_samples", "utt2dataset")
    }
    metadata = {}
    for source in args.inputs:
        utterances = None
        for name in ("wav.scp", "utt2lang", "utt2num_samples"):
            rows = {}
            for line in (source / name).read_text().splitlines():
                key, value = line.split(maxsplit=1)
                if key in rows or key in merged[name]:
                    raise ValueError(f"Duplicate utterance {key} in {source / name}")
                rows[key] = value
            if utterances is not None and utterances != rows.keys():
                raise ValueError(f"Mismatched utterances in {source / name}")
            utterances = rows.keys()
            merged[name].update(rows)
        merged["utt2dataset"].update({key: source.name for key in utterances})
        for name in ("audio_format", "feats_type"):
            value = (source / name).read_text()
            if name in metadata and metadata[name] != value:
                raise ValueError(f"Inconsistent {name} in {source}")
            metadata[name] = value
    target = args.output_dir
    if target.resolve() in [source.resolve() for source in args.inputs]:
        parser.error("Output must not overwrite an input directory.")
    target.mkdir(parents=True, exist_ok=True)
    for name, rows in merged.items():
        (target / name).write_text(
            "".join(f"{key} {value}\n" for key, value in sorted(rows.items()))
        )
    for name, value in metadata.items():
        (target / name).write_text(value)
    for source, dest in (("utt2lang", "lang2utt"), ("utt2dataset", "dataset2utt")):
        groups = defaultdict(list)
        for key, value in merged[source].items():
            groups[value].append(key)
        text = "".join(
            f"{value} {' '.join(sorted(keys))}\n"
            for value, keys in sorted(groups.items())
        )
        (target / dest).write_text(text)
        if dest == "lang2utt":
            (target / "category2utt").write_text(text)


if __name__ == "__main__":
    main()
