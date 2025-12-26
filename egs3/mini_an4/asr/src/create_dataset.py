"""Prepare the Mini AN4 dataset for ESPnet3 ASR recipes (simplified)."""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from espnet3.utils.download import extract_targz, setup_logger

LOGGER = logging.getLogger(__name__)

SPH_DIRS = {"train": "an4_clstk", "test": "an4test_clstk"}
TRANSCRIPTS = {"train": "an4_train.transcription", "test": "an4_test.transcription"}

# "<s> WORDS </s> (a-b-c)" を想定
LINE_RE = re.compile(r"^(?P<words>.+?)\s+\((?P<src>[^)]+)\)\s*$")


@dataclass(frozen=True)
class Entry:
    utt_id: str
    wav_path: Path
    text: str


def is_gzip(path: Path) -> bool:
    return path.is_file() and path.read_bytes()[:2] == b"\x1f\x8b"


def ensure_extracted(archive: Path, dataset_dir: Path, *, logger: logging.Logger) -> Path:
    downloads_dir = dataset_dir / "downloads"
    an4_root = downloads_dir / "an4"
    if an4_root.exists():
        logger.info("Found existing downloads: %s", an4_root)
        return an4_root

    if not archive.is_file():
        raise FileNotFoundError(f"Archive not found: {archive}")
    if not is_gzip(archive):
        raise RuntimeError(f"Archive is not gzip: {archive}")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    extract_targz(archive, dataset_dir, logger=logger)

    if not an4_root.exists():
        raise RuntimeError(f"AN4 root not found after extract: {an4_root}")
    return an4_root


def parse_line(line: str) -> tuple[str, str, str]:
    m = LINE_RE.match(line)
    if not m:
        raise ValueError(f"Malformed transcript line: {line}")

    words = m.group("words")
    # transcript の <s> </s> を雑に剥がす（現状仕様踏襲）
    if words.startswith("<s> "):
        words = words[4:]
    if words.endswith(" </s>"):
        words = words[:-5]

    src = m.group("src")  # 例: abc-def-ghi
    pre, mid, last = src.split("-")
    utt_id = f"{mid}-{pre}-{last}"
    return utt_id, src, words


def load_entries(an4_root: Path, split: str) -> list[tuple[str, str, str]]:
    path = an4_root / "etc" / TRANSCRIPTS[split]
    if not path.is_file():
        raise FileNotFoundError(f"Transcript not found: {path}")

    entries: list[tuple[str, str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(parse_line(line))

    entries.sort(key=lambda x: x[0])
    return entries


def sph_to_wav(sph2pipe: str, sph: Path, wav: Path) -> None:
    if wav.exists():
        return
    wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sph2pipe, "-f", "wav", "-p", "-c", "1", str(sph)]
    with wav.open("wb") as f:
        subprocess.run(cmd, stdout=f, check=True)


def prepare_split(an4_root: Path, dataset_dir: Path, split: str, sph2pipe: str) -> list[Entry]:
    wav_dir = dataset_dir / "wav" / split
    sph_dir = an4_root / "wav" / SPH_DIRS[split]

    out: list[Entry] = []
    for utt_id, src, text in load_entries(an4_root, split):
        mid = src.split("-")[1]
        sph = sph_dir / mid / f"{src}.sph"
        if not sph.is_file():
            raise FileNotFoundError(f"Missing sph file: {sph}")

        wav = (wav_dir / f"{utt_id}.wav").resolve()
        sph_to_wav(sph2pipe, sph, wav)
        out.append(Entry(utt_id=utt_id, wav_path=wav, text=text))
    return out


def write_manifest(path: Path, entries: Iterable[Entry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(f"{e.utt_id}\t{e.wav_path}\t{e.text}\n")


def create_dataset(dataset_dir: Path, *, archive_path: Path | None = None) -> None:
    dataset_dir = Path(dataset_dir)
    logger = setup_logger(name="mini_an4.create_dataset")

    archive = Path("../../../egs2/mini_an4/asr1/downloads.tar.gz")
    an4_root = ensure_extracted(archive, dataset_dir, logger=logger)

    sph2pipe = shutil.which("sph2pipe")
    if not sph2pipe:
        raise RuntimeError("sph2pipe not found in PATH. Please install it.")

    train = prepare_split(an4_root, dataset_dir, "train", sph2pipe)
    test = prepare_split(an4_root, dataset_dir, "test", sph2pipe)

    if len(train) < 2:
        raise RuntimeError("Training data is too small (need >= 2 for dev+nodev).")

    manifest_dir = dataset_dir / "manifest"
    write_manifest(manifest_dir / "train_dev.tsv", train[:1])
    write_manifest(manifest_dir / "train_nodev.tsv", train[1:])
    write_manifest(manifest_dir / "test.tsv", test)

    LOGGER.info("Prepared Mini AN4 manifests under %s", manifest_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Prepare Mini AN4 dataset")
    p.add_argument("--dataset_dir", type=Path, required=True)
    p.add_argument("--archive_path", type=Path, default=None)
    a = p.parse_args()
    create_dataset(a.dataset_dir, archive_path=a.archive_path)
