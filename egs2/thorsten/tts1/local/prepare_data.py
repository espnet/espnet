#!/usr/bin/env python3

"""Prepare Thorsten-Voice 2022.10 for ESPnet TTS."""

import argparse
import hashlib
from pathlib import Path

METADATA_SPLITS = ("train", "dev", "test")


def load_metadata(db_root: Path) -> dict[str, str]:
    """Return {utterance_id: normalized_text} from all released metadata."""

    utterances = {}

    for split in METADATA_SPLITS:
        metadata = db_root / f"metadata_{split}.csv"

        if not metadata.is_file():
            raise FileNotFoundError(f"Missing metadata file: {metadata}")

        for line_no, line in enumerate(
            metadata.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line:
                continue

            fields = line.split("|", 2)
            if len(fields) != 3:
                raise ValueError(f"Malformed line {metadata}:{line_no}: {line!r}")

            utt_id, _, normalized_text = fields
            utt_id = utt_id.strip()
            normalized_text = normalized_text.strip()

            if utt_id in utterances:
                raise ValueError(f"Duplicate utterance ID: {utt_id}")

            utterances[utt_id] = normalized_text

    return utterances


def validate_audio(db_root: Path, utterances: dict[str, str]) -> None:
    """Check that every metadata entry has audio and report unused WAVs."""

    wav_dir = db_root / "wavs"

    if not wav_dir.is_dir():
        raise FileNotFoundError(f"Missing WAV directory: {wav_dir}")

    wav_ids = {p.stem for p in wav_dir.glob("*.wav")}
    metadata_ids = set(utterances)

    missing = metadata_ids - wav_ids
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} metadata entries have no WAV file: "
            f"{', '.join(sorted(missing)[:10])}"
        )

    orphan = wav_ids - metadata_ids
    if orphan:
        print(f"Info: ignoring {len(orphan)} WAV files without metadata.")


def split_utterances(
    utterances: dict[str, str],
    valid_size: int,
    test_size: int,
    seed: int,
) -> dict[str, list[str]]:
    """Create a deterministic train/valid/test split."""

    if valid_size <= 0 or test_size <= 0:
        raise ValueError("valid_size and test_size must be positive")

    if len(utterances) <= valid_size + test_size:
        raise ValueError("Dataset is too small for requested valid/test sizes")

    def split_key(utt_id: str) -> bytes:
        return hashlib.sha256(f"{seed}:{utt_id}".encode()).digest()

    ids = sorted(utterances, key=split_key)

    valid = ids[:valid_size]
    test = ids[valid_size : valid_size + test_size]
    train = ids[valid_size + test_size :]

    return {
        "train": sorted(train),
        "valid": sorted(valid),
        "test": sorted(test),
    }


def write_split(
    db_root: Path,
    output_dir: Path,
    utt_ids: list[str],
    utterances: dict[str, str],
    speaker: str,
) -> None:
    """Write one ESPnet/Kaldi-style data directory."""

    output_dir.mkdir(parents=True, exist_ok=True)

    with (
        (output_dir / "wav.scp").open("w", encoding="utf-8") as wav_scp,
        (output_dir / "text").open("w", encoding="utf-8") as text,
        (output_dir / "utt2spk").open("w", encoding="utf-8") as utt2spk,
    ):
        for utt_id in utt_ids:
            wav_scp.write(f"{utt_id} {db_root}/wavs/{utt_id}.wav\n")
            text.write(f"{utt_id} {utterances[utt_id]}\n")
            utt2spk.write(f"{utt_id} {speaker}\n")


def prepare_dataset(
    db_root: Path,
    output_root: Path,
    valid_size: int = 100,
    test_size: int = 100,
    seed: int = 100,
    speaker: str = "thorsten",
) -> dict[str, list[str]]:
    utterances = load_metadata(db_root)
    validate_audio(db_root, utterances)

    splits = split_utterances(
        utterances,
        valid_size=valid_size,
        test_size=test_size,
        seed=seed,
    )

    for split, utt_ids in splits.items():
        write_split(
            db_root,
            output_root / split,
            utt_ids,
            utterances,
            speaker,
        )

    print(f"Prepared {len(utterances)} transcribed utterances:")
    for split in ("train", "valid", "test"):
        print(f"  {split}: {len(splits[split])}")

    return splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--valid-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--speaker", default="thorsten")
    args = parser.parse_args()

    prepare_dataset(
        db_root=args.db_root,
        output_root=args.output_root,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
        speaker=args.speaker,
    )


if __name__ == "__main__":
    main()
