"""Prepare the Mini AN4 dataset for ESPnet3 TTS recipes."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Iterable

from egs3.mini_an4.asr.src.creating_dataset import (
    Entry,
    ensure_extracted,
    prepare_split,
    write_manifest,
)
from espnet3.utils.download_utils import setup_logger

LOGGER = logging.getLogger(__name__)


def _write_token_list(path: Path, entries: Iterable[Entry]) -> None:
    chars = sorted({char for entry in entries for char in entry.text if char != " "})
    tokens = ["<unk>", "<space>", *chars]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(tokens) + "\n", encoding="utf-8")


def create_dataset(dataset_dir: Path, archive_path: Path | None = None) -> None:
    dataset_dir = Path(dataset_dir)
    archive = Path(archive_path) if archive_path is not None else None
    if archive is None:
        archive = Path("../../../egs2/mini_an4/asr1/downloads.tar.gz")

    logger = setup_logger(name="mini_an4.tts.create_dataset")
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
    _write_token_list(dataset_dir / "tokens.txt", train)

    LOGGER.info("Prepared Mini AN4 TTS manifests under %s", manifest_dir)
