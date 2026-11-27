#!/usr/bin/env python3
"""Download google/fleurs (all 102 configs) and write TSV manifests.

The audio bytes are written to <out_root>/audio/<config>/<split>/<id>_<stem>.wav
so that the manifests point to plain files. The columns are read by
local/prepare_fleurs_cs_lid_data.py.
"""

import argparse
import csv
import os
from pathlib import Path

FIELDS = ["id", "lang_id_name", "path", "source_file", "duration"]
SPLITS = {"train": "train", "validation": "dev", "test": "test"}


def main() -> None:
    """Download every FLEURS config and write {train,dev,test}.tsv."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, default=Path("downloads/cache"))
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()

    from datasets import Audio, load_dataset
    from prepare_fleurs_cs_lid_data import FLEURS_TO_ISO3

    configs = list(FLEURS_TO_ISO3)
    audio_root = args.out_root / "audio"
    args.out_root.mkdir(parents=True, exist_ok=True)
    tmp_paths = {
        s: args.out_root / f".{n}.tsv.{os.getpid()}" for s, n in SPLITS.items()
    }
    files = {s: p.open("w", encoding="utf-8", newline="") for s, p in tmp_paths.items()}
    writers = {
        s: csv.DictWriter(f, delimiter="\t", fieldnames=FIELDS)
        for s, f in files.items()
    }
    for writer in writers.values():
        writer.writeheader()
    done = False
    try:
        for i, config in enumerate(configs, 1):
            print(f"google/fleurs {config} ({i}/{len(configs)})", flush=True)
            for split in SPLITS:
                dataset = load_dataset(
                    "google/fleurs",
                    config,
                    revision=args.revision,
                    split=split,
                    streaming=True,
                    cache_dir=str(args.cache_dir),
                ).cast_column("audio", Audio(decode=False))
                for row in dataset:
                    audio = row["audio"]
                    source = row.get("path") or audio.get("path") or str(row["id"])
                    stem = Path(source).stem
                    dest = audio_root / config / split / f"{row['id']}_{stem}.wav"
                    if not dest.is_file():
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        tmp = dest.with_name(f".{dest.name}.{os.getpid()}")
                        tmp.write_bytes(audio["bytes"])
                        os.replace(tmp, dest)
                    sampling_rate = int(row.get("sampling_rate") or 16000)
                    duration = int(row["num_samples"]) / sampling_rate
                    writers[split].writerow(
                        {
                            "id": row["id"],
                            "lang_id_name": config,
                            "path": str(dest.resolve()),
                            "source_file": Path(source).name,
                            "duration": f"{duration:.9f}",
                        }
                    )
        done = True
    finally:
        for f in files.values():
            f.close()
        for split, name in SPLITS.items():
            if done:
                os.replace(tmp_paths[split], args.out_root / f"{name}.tsv")
            else:
                tmp_paths[split].unlink(missing_ok=True)
    print(f"wrote {args.out_root}/{{train,dev,test}}.tsv")


if __name__ == "__main__":
    main()
