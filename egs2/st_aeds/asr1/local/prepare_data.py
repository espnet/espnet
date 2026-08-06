#!/usr/bin/env python3
"""Prepare ST-AEDS / OpenSLR SLR45 as Kaldi-style ESPnet data directories."""

from __future__ import annotations

import argparse
import collections
from pathlib import Path
import re
import shutil
import sys
import wave


SPEAKER_RE = re.compile(r"^([fm][0-9]{4})_")


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def read_transcripts(root: Path):
    text_path = root / "text.txt"
    if not text_path.is_file():
        raise FileNotFoundError(f"Missing transcript file: {text_path}")

    entries = []
    missing_audio = []
    empty_transcripts = []
    seen = set()
    with text_path.open("r", encoding="utf-8") as stream:
        for line_no, raw_line in enumerate(stream, 1):
            line = raw_line.rstrip("\n\r")
            if not line.strip():
                continue
            if "\t" not in line:
                raise RuntimeError(f"{text_path}:{line_no}: expected '<wav>\\t<text>'")
            wav_name, text = line.split("\t", 1)
            wav_name = wav_name.strip()
            transcript = normalize_text(text)
            if not wav_name:
                raise RuntimeError(f"{text_path}:{line_no}: empty wav name")
            if wav_name in seen:
                raise RuntimeError(
                    f"{text_path}:{line_no}: duplicate wav entry: {wav_name}"
                )
            seen.add(wav_name)

            wav_path = root / wav_name
            if not wav_path.is_file():
                missing_audio.append(wav_name)
                continue
            if not transcript:
                empty_transcripts.append(wav_name)
                continue
            if wav_path.suffix.lower() != ".wav":
                raise RuntimeError(
                    f"{text_path}:{line_no}: expected .wav audio: {wav_name}"
                )

            match = SPEAKER_RE.match(wav_name)
            if match is None:
                raise RuntimeError(
                    f"{text_path}:{line_no}: cannot infer speaker from {wav_name}"
                )
            utt_id = wav_path.stem
            spk_id = match.group(1)
            entries.append((spk_id, utt_id, wav_path.resolve(), transcript))

    wavs = {path.name for path in root.glob("*.wav")}
    missing_transcripts = sorted(wavs - seen)
    if missing_transcripts:
        raise RuntimeError(
            "Missing transcript entries for wav files: "
            + ", ".join(missing_transcripts[:10])
        )

    return (
        sorted(entries, key=lambda item: (item[0], item[1])),
        missing_audio,
        empty_transcripts,
    )


def split_entries(entries, dev_per_spk: int, test_per_spk: int):
    by_spk = collections.defaultdict(list)
    for entry in entries:
        by_spk[entry[0]].append(entry)

    splits = {"train": [], "dev": [], "test": []}
    for spk_id in sorted(by_spk):
        speaker_entries = sorted(by_spk[spk_id], key=lambda item: item[1])
        required = dev_per_spk + test_per_spk + 1
        if len(speaker_entries) < required:
            raise RuntimeError(
                f"Speaker {spk_id} has {len(speaker_entries)} utterances; "
                f"need at least {required}"
            )
        splits["train"].extend(speaker_entries[: -(dev_per_spk + test_per_spk)])
        splits["dev"].extend(
            speaker_entries[-(dev_per_spk + test_per_spk) : -test_per_spk]
        )
        splits["test"].extend(speaker_entries[-test_per_spk:])

    return splits


def num_samples(wav_path: Path) -> int:
    with wave.open(str(wav_path), "rb") as wav_file:
        return wav_file.getnframes()


def write_split(name: str, entries, data_dir: Path) -> None:
    split_dir = data_dir / name
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True)

    with (split_dir / "text").open("w", encoding="utf-8") as text_f, (
        split_dir / "wav.scp"
    ).open("w", encoding="utf-8") as wav_f, (split_dir / "utt2spk").open(
        "w", encoding="utf-8"
    ) as utt2spk_f, (split_dir / "utt2num_samples").open(
        "w", encoding="utf-8"
    ) as samples_f:
        for spk_id, utt_id, wav_path, transcript in sorted(
            entries, key=lambda item: item[1]
        ):
            print(f"{utt_id} {transcript}", file=text_f)
            print(f"{utt_id} {wav_path}", file=wav_f)
            print(f"{utt_id} {spk_id}", file=utt2spk_f)
            print(f"{utt_id} {num_samples(wav_path)}", file=samples_f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Extracted ST-AEDS root")
    parser.add_argument("--data-dir", default="data", help="Output data directory")
    parser.add_argument("--dev-per-spk", type=int, default=40)
    parser.add_argument("--test-per-spk", type=int, default=40)
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    data_dir = Path(args.data_dir)
    if not root.is_dir():
        parser.error(f"root does not exist or is not a directory: {root}")
    if args.dev_per_spk <= 0 or args.test_per_spk <= 0:
        parser.error("--dev-per-spk and --test-per-spk must be positive")

    entries, missing_audio, empty_transcripts = read_transcripts(root)
    if not entries:
        raise RuntimeError(f"No usable ST-AEDS entries found under {root}")

    splits = split_entries(entries, args.dev_per_spk, args.test_per_spk)
    for name in ("train", "dev", "test"):
        write_split(name, splits[name], data_dir)

    speakers = sorted({entry[0] for entry in entries})
    print(f"Prepared ST-AEDS data from {root}")
    print(f"Speakers: {len(speakers)} ({', '.join(speakers)})")
    if missing_audio:
        print(
            "Warning: skipped transcript entries with missing audio: "
            + ", ".join(missing_audio[:10]),
            file=sys.stderr,
        )
    if empty_transcripts:
        print(
            "Warning: skipped audio entries with empty transcripts: "
            + ", ".join(empty_transcripts[:10]),
            file=sys.stderr,
        )
    for name in ("train", "dev", "test"):
        print(f"{name}: {len(splits[name])} utterances")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
