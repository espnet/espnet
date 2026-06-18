#!/usr/bin/env python3

import argparse
import csv
import os
import random
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg")


def normalize_label(label):
    return label.strip().replace(" ", "_")


def normalize_uttid(value):
    keep = []
    for char in value:
        if char.isalnum() or char in ("-", "_", "."):
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("._-")


def sniff_dialect(path):
    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t")
    except csv.Error:
        return csv.excel


def has_header(path, dialect):
    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(4096)
    first_line = sample.splitlines()[0] if sample.splitlines() else ""
    columns = {item.strip().lower() for item in first_line.split(dialect.delimiter)}
    known_columns = {
        "audio",
        "audio_path",
        "wav",
        "wav_path",
        "path",
        "filename",
        "file",
        "video_id",
        "youtube_id",
        "ytid",
        "id",
        "label",
        "class",
        "category",
        "caption",
        "start",
        "start_time",
        "start_sec",
        "timestamp",
    }
    if columns & known_columns:
        return True
    try:
        return csv.Sniffer().has_header(sample)
    except csv.Error:
        return False


def get_first(row, names):
    for name in names:
        if name in row and row[name] is not None and row[name].strip():
            return row[name].strip()
    return None


def count_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def progress(iterable, desc, total=None):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, unit="rows")


def rows_from_metadata(path):
    dialect = sniff_dialect(path)
    header = has_header(path, dialect)
    total = count_lines(path)

    with open(path, "r", encoding="utf-8") as f:
        if header:
            reader = csv.DictReader(f, dialect=dialect)
            for row in progress(reader, f"Reading {path.name}", max(total - 1, 0)):
                audio = get_first(
                    row,
                    (
                        "audio",
                        "audio_path",
                        "wav",
                        "wav_path",
                        "path",
                        "filename",
                        "file",
                    ),
                )
                video_id = get_first(row, ("video_id", "youtube_id", "ytid", "id"))
                label = get_first(row, ("label", "class", "category", "caption"))
                start = get_first(
                    row, ("start", "start_time", "start_sec", "timestamp")
                )
                split = get_first(row, ("split", "subset", "set"))

                if label is None:
                    raise RuntimeError(
                        f"{path} has a header but no label/class/category column"
                    )
                yield {
                    "audio": audio,
                    "video_id": video_id,
                    "start": start,
                    "label": label,
                    "split": split,
                }
        else:
            reader = csv.reader(f, dialect=dialect)
            for row in progress(reader, f"Reading {path.name}", total):
                if not row or row[0].startswith("#"):
                    continue
                row = [item.strip() for item in row]
                if len(row) >= 4:
                    # Common VGGSound format: youtube_id, start_seconds, label, split
                    yield {
                        "audio": None,
                        "video_id": row[0],
                        "start": row[1],
                        "label": row[2],
                        "split": row[3],
                    }
                elif len(row) >= 3:
                    # VGGSound-like format without split: youtube_id, start_seconds, label
                    yield {
                        "audio": None,
                        "video_id": row[0],
                        "start": row[1],
                        "label": row[2],
                        "split": None,
                    }
                elif len(row) == 2:
                    yield {
                        "audio": row[0],
                        "video_id": None,
                        "start": None,
                        "label": row[1],
                        "split": None,
                    }
                else:
                    raise RuntimeError(f"Unsupported row in {path}: {row}")


def candidate_audio_paths(root, split, item):
    if item["audio"]:
        path = Path(item["audio"])
        if path.is_absolute():
            yield path
        else:
            yield root / path
            yield root / split / path
            yield root / "audio" / split / path
            yield root / "audio" / path

    if item["video_id"]:
        base = normalize_uttid(item["video_id"])
        if item["start"]:
            start = normalize_uttid(item["start"])
            try:
                start_int = int(float(item["start"]))
                start_padded = f"{start_int:06d}"
            except ValueError:
                start_padded = start
            bases = (
                f"{base}_{start_padded}",
                f"{base}_{start}",
                f"{base}-{start}",
                f"{base}_{start}000",
                base,
            )
        else:
            bases = (base,)

        dirs = (
            root / split,
            root / "audio" / split,
            root / "audio",
            root / "wav" / split,
            root / "wav",
            root,
        )
        for directory in dirs:
            for stem in bases:
                for ext in AUDIO_EXTS:
                    yield directory / f"{stem}{ext}"


def find_audio(root, split, item):
    seen = set()
    for path in candidate_audio_paths(root, split, item):
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            return path
    return None


def infer_metadata(root, split):
    candidates = (
        root / f"{split}.csv",
        root / f"{split}.tsv",
        root / f"vggsound_{split}.csv",
        root / f"vggsound_{split}.tsv",
        root / "metadata" / f"{split}.csv",
        root / "metadata" / f"{split}.tsv",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find metadata for split "
        f"{split}. Expected one of: {', '.join(str(p) for p in candidates)}"
    )


def infer_single_metadata(root):
    candidates = (
        root / "vggsound.csv",
        root / "vggsound.tsv",
        root / "metadata" / "vggsound.csv",
        root / "metadata" / "vggsound.tsv",
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def read_items(metadata):
    return list(rows_from_metadata(metadata))


def split_items_from_single_metadata(items, valid_ratio, valid_seed):
    train_items = []
    test_items = []
    for item in progress(items, "Splitting metadata", len(items)):
        split = (item.get("split") or "").strip().lower()
        if split == "train":
            train_items.append(item)
        elif split in ("test", "eval"):
            test_items.append(item)
        else:
            raise RuntimeError(
                "Single metadata mode expects split values train/test. "
                f"Got split={item.get('split')!r} for item={item}"
            )

    rng = random.Random(valid_seed)
    indices = list(range(len(train_items)))
    rng.shuffle(indices)
    valid_size = max(1, int(round(len(indices) * valid_ratio)))
    valid_indices = set(indices[:valid_size])

    valid_items = []
    kept_train_items = []
    for index, item in enumerate(
        progress(train_items, "Selecting validation split", len(train_items))
    ):
        if index in valid_indices:
            valid_items.append(item)
        else:
            kept_train_items.append(item)

    print(
        "Metadata split before audio filtering: "
        f"train={len(kept_train_items)}, valid={len(valid_items)}, "
        f"test={len(test_items)}"
    )

    return {
        "train": kept_train_items,
        "valid": valid_items,
        "test": test_items,
    }


def write_split(root, output_root, split, items, metadata):
    out_dir = output_root / split
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = 0
    written = 0
    with (
        open(out_dir / "wav.scp", "w", encoding="utf-8") as wav_f,
        open(out_dir / "text", "w", encoding="utf-8") as text_f,
        open(out_dir / "utt2spk", "w", encoding="utf-8") as utt2spk_f,
    ):
        for index, item in enumerate(progress(items, f"Writing {split}", len(items))):
            audio_path = find_audio(root, split, item)
            if audio_path is None:
                missing += 1
                continue

            source_id = item["video_id"] or Path(item["audio"]).stem
            start = f"-{normalize_uttid(item['start'])}" if item["start"] else ""
            uttid = f"vggsound-{split}-{normalize_uttid(source_id)}{start}-{index:06d}"
            label = normalize_label(item["label"])

            print(f"{uttid} {audio_path}", file=wav_f)
            print(f"{uttid} {label}", file=text_f)
            print(f"{uttid} dummy", file=utt2spk_f)
            written += 1

    print(
        f"{split}: wrote {written} utterances from {metadata}; "
        f"skipped {missing} rows with missing audio"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("vggsound_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--train-metadata", type=Path)
    parser.add_argument("--valid-metadata", type=Path)
    parser.add_argument("--test-metadata", type=Path)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--valid-seed", type=int, default=0)
    args = parser.parse_args()

    root = args.vggsound_root
    output_root = args.output_root

    single_metadata = args.metadata or infer_single_metadata(root)
    if single_metadata is not None and not (
        args.train_metadata or args.valid_metadata or args.test_metadata
    ):
        split_to_items = split_items_from_single_metadata(
            read_items(single_metadata), args.valid_ratio, args.valid_seed
        )
        for split, items in split_to_items.items():
            write_split(root, output_root, split, items, single_metadata)
    else:
        split_to_metadata = {
            "train": args.train_metadata or infer_metadata(root, "train"),
            "valid": args.valid_metadata or infer_metadata(root, "valid"),
            "test": args.test_metadata or infer_metadata(root, "test"),
        }
        for split, metadata in split_to_metadata.items():
            write_split(root, output_root, split, read_items(metadata), metadata)


if __name__ == "__main__":
    main()
