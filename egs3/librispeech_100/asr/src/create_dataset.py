"""LibriSpeech downloader for the egs3 LibriSpeech 100h ASR recipe.

This module is used in two ways:
  1) As an ESPnet3 stage function via config (`create_dataset.func`), where the
     caller invokes :func:`create_dataset` directly with keyword arguments.
  2) As a standalone script (`python src/create_dataset.py ...`) for quick manual
     downloads during development.

The implementation intentionally only downloads/extracts the original LibriSpeech
archives into the specified directory (no preprocessing, no dataset conversion).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from espnet3.utils.download import download_url, extract_targz, setup_logger

OPENSLR_BASE_URL = "https://www.openslr.org/resources/12"

SPLITS = {
    "train.clean.100": "train-clean-100.tar.gz",
    "validation.clean": "dev-clean.tar.gz",
    "validation.other": "dev-other.tar.gz",
    "test.clean": "test-clean.tar.gz",
    "test.other": "test-other.tar.gz",
}


def download_and_extract_if_needed(
    *,
    split: str,
    url: str,
    dataset_dir: Path,
    extracted_dir: Path,
    archive_name: str,
    logger,
    step_percent: int = 5,
) -> None:
    """Download and extract one split if not already present."""
    if extracted_dir.exists():
        logger.info(f"Skip split '{split}' (already exists): {extracted_dir}")
        return

    dataset_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dataset_dir / archive_name

    logger.info(f"Start processing split: {split}")

    if not archive_path.exists():
        download_url(
            url=url,
            dst_path=archive_path,
            logger=logger,
            step_percent=step_percent,
        )
    else:
        logger.info(f"Archive exists, skip download: {archive_name}")

    extract_targz(archive_path, dataset_dir, logger)
    logger.info(f"Finished split: {split}")


def create_dataset(
    dataset_dir: Path,
    *,
    step_percent: int = 5,
    splits: list[str] | None = None,
) -> None:
    """Download (and extract) requested LibriSpeech splits into ``dataset_dir``.

    Args:
        dataset_dir: Destination directory. Archives are stored under this
            directory and extracted into ``dataset_dir / "LibriSpeech"``.
        step_percent: Logging granularity for download progress.
        splits: Optional subset of keys from :data:`SPLITS`. If omitted, all
            splits in :data:`SPLITS` are processed.
    """
    logger = setup_logger(
        name="create_dataset",
        log_dir=dataset_dir / "logs",
    )

    librispeech_root = dataset_dir / "LibriSpeech"

    requested = list(SPLITS.keys()) if splits is None else list(splits)
    unknown = [s for s in requested if s not in SPLITS]
    if unknown:
        raise ValueError(f"Unknown split(s): {unknown}. Valid: {list(SPLITS.keys())}")

    for split in requested:
        filename = SPLITS[split]
        extracted_dir = librispeech_root / filename.replace(".tar.gz", "")
        url = f"{OPENSLR_BASE_URL}/{filename}"

        download_and_extract_if_needed(
            split=split,
            url=url,
            dataset_dir=dataset_dir,
            extracted_dir=extracted_dir,
            archive_name=filename,
            logger=logger,
            step_percent=step_percent,
        )

    logger.info("All requested splits processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download LibriSpeech to a directory.")
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="Target directory to store archives and extracted files.",
    )
    parser.add_argument(
        "--step_percent",
        type=int,
        default=5,
        help="Progress logging granularity (percent per log).",
    )

    args = parser.parse_args()
    create_dataset(
        args.dataset_dir,
        step_percent=args.step_percent,
    )
