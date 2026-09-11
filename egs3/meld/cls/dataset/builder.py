"""MELD dataset builder."""

from __future__ import annotations

import csv
import logging
import os
import re
import shutil
import subprocess
import urllib.error
from importlib import resources
from pathlib import Path
from typing import Iterator

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.download_utils import download_url, extract_targz

logger = logging.getLogger(__name__)

NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9]+")


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()


def _normalize_speaker(name: str) -> str:
    """Normalize a speaker name so it is safe to embed in an utterance id."""
    return NON_ALNUM_RE.sub("_", name).strip("_")


def _utterance_id(row: dict, split: str) -> str:
    """Build the utterance id for one annotation row."""
    speaker = _normalize_speaker(row["Speaker"])
    return (
        f"{speaker}"
        f"-dia{row['Dialogue_ID']}"
        f"-utt{row['Utterance_ID']}"
        f"-sea{row['Season']}"
        f"-epi{row['Episode']}"
        f"-{split}"
    )


def iter_source_candidates(
    recipe_root: Path,
    source_dir: str | Path | None = None,
) -> Iterator[Path]:
    """Yield candidate MELD source roots in priority order."""
    if source_dir is not None:
        yield Path(source_dir).expanduser()

    env_path = os.environ.get(str(_CFG["source_env_var"]))
    if env_path:
        yield Path(env_path).expanduser()

    yield recipe_root / _CFG["dataset_path"]


def missing_source_entries(source_root: Path) -> list[str]:
    """Return required source paths that are absent from ``source_root``."""
    missing: list[str] = []
    for spec in _CFG["splits"].values():
        annotation = source_root / _CFG["metadata_subdir"] / spec["csv_name"]
        if not annotation.is_file():
            missing.append(str(annotation))
        clips = source_root / _CFG["audio_subdir"] / spec["audio_subdir"]
        if not clips.is_dir():
            missing.append(str(clips))
    return missing


def resolve_source_root(
    recipe_root: Path,
    source_dir: str | Path | None = None,
) -> Path:
    """Resolve the usable MELD source root for this recipe.

    Args:
        recipe_root: Recipe root directory.
        source_dir: Optional explicit source root that wins over the
            environment variable and the in-recipe download directory.

    Returns:
        Path to a directory holding both the annotation CSVs and the clips.

    Raises:
        FileNotFoundError: If no candidate directory holds a complete corpus.
    """
    checked: list[str] = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        checked.append(str(candidate))
        if not missing_source_entries(candidate):
            return candidate

    env_var = str(_CFG["source_env_var"])
    raise FileNotFoundError(
        "MELD source not found. Checked these locations:\n"
        + "\n".join(f"  - {path}" for path in checked)
        + "\n"
        + f"Place the corpus under <recipe_dir>/{_CFG['dataset_path']} "
        + f"or set {env_var} to the dataset root. The directory must contain "
        + f"{_CFG['metadata_subdir']}/<split>_sent_emo.csv and "
        + f"{_CFG['audio_subdir']}/<split>/."
    )


def resolve_data_root(recipe_root: Path) -> Path:
    """Resolve where converted audio and manifests are written."""
    env_var = _CFG.get("output_env_var")
    if env_var:
        env_path = os.environ.get(str(env_var))
        if env_path:
            return Path(env_path).expanduser()
    return recipe_root / _CFG["data_path"]


def resolve_download_root(
    recipe_root: Path,
    source_dir: str | Path | None = None,
) -> Path:
    """Resolve where the corpus is downloaded when it is missing.

    Unlike :func:`resolve_source_root` this never fails: it returns the first
    candidate location regardless of whether the corpus is already there.
    """
    return next(iter_source_candidates(recipe_root, source_dir))


def _download_archive(destination: Path) -> None:
    """Download the MELD archive, trying each configured URL in order.

    Raises:
        RuntimeError: If every URL fails.
    """
    archive_name = str(_CFG["archive_name"])
    # Download into a part file so an interrupted run cannot leave a truncated
    # archive that later looks complete.
    part = destination.with_suffix(destination.suffix + ".part")
    errors: list[str] = []
    for url_base in _CFG["data_urls"]:
        url = f"{str(url_base).rstrip('/')}/{archive_name}"
        logger.info("Downloading MELD (about 11 GB) from %s", url)
        try:
            download_url(url, part, logger=logger)
        except (urllib.error.URLError, OSError) as exc:
            logger.warning("Download failed from %s: %s", url, exc)
            errors.append(f"  - {url}: {exc}")
            part.unlink(missing_ok=True)
            continue
        part.rename(destination)
        return

    raise RuntimeError(
        "Failed to download "
        + archive_name
        + " from every configured URL:\n"
        + "\n".join(errors)
    )


def _move_first_existing(candidates: list[Path], target: Path) -> None:
    """Move the first existing candidate to ``target``.

    Raises:
        RuntimeError: If none of the candidates exist.
    """
    for candidate in candidates:
        if candidate.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            candidate.rename(target)
            return
    raise RuntimeError(
        "Expected file missing from the MELD archive: "
        + ", ".join(str(path) for path in candidates)
    )


def _unpack_archive(source_root: Path, archive: Path) -> None:
    """Unpack the MELD archive into the layout this recipe expects.

    The distributed archive nests one tar per split and names the splits
    differently from the annotation CSVs, so the entries are renamed to
    ``<audio_subdir>/{train,valid,test}`` and the CSVs are collected under
    ``<metadata_subdir>/``.
    """
    raw_dir = source_root / "MELD.Raw"
    extract_targz(archive, source_root, logger=logger)

    split_archives = (
        ("dev.tar.gz", "dev_splits_complete", "valid"),
        ("test.tar.gz", "output_repeated_splits_test", "test"),
        ("train.tar.gz", "train_splits", "train"),
    )
    for archive_name, extracted_name, split in split_archives:
        extract_targz(raw_dir / archive_name, source_root, logger=logger)
        _move_first_existing(
            [source_root / extracted_name, raw_dir / extracted_name],
            raw_dir / split,
        )

    metadata_dir = source_root / _CFG["metadata_subdir"]
    metadata_dir.mkdir(parents=True, exist_ok=True)
    csv_moves = (
        ("test_sent_emo.csv", "test_sent_emo.csv"),
        ("dev_sent_emo.csv", "valid_sent_emo.csv"),
        ("train_sent_emo.csv", "train_sent_emo.csv"),
    )
    for original_name, target_name in csv_moves:
        _move_first_existing(
            [raw_dir / original_name, source_root / original_name],
            metadata_dir / target_name,
        )

    readme = raw_dir / "README.txt"
    if readme.exists():
        readme.rename(source_root / "README.txt")

    if _CFG.get("remove_archive", False):
        archive.unlink(missing_ok=True)
        for inner in raw_dir.glob("*.tar.gz"):
            inner.unlink()

    raw_dir.rename(source_root / _CFG["audio_subdir"])


class MELDBuilder(DatasetBuilder):
    """Prepare and build MELD assets for ESPnet3 recipes."""

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether a complete MELD source tree is reachable."""
        recipe_root = Path(recipe_dir).resolve()
        try:
            resolve_source_root(recipe_root, source_dir=source_dir)
        except FileNotFoundError:
            return False
        return True

    def prepare_source(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> None:
        """Download and unpack MELD unless the corpus is already available.

        Args:
            recipe_dir: Recipe root directory.
            source_dir: Optional explicit source root that wins over the
                environment variable and the in-recipe download directory.
            **_kwargs: Unused extra options for API compatibility.

        Returns:
            None.

        Raises:
            RuntimeError: If the download fails, the archive layout is
                unexpected, or the unpacked tree is still incomplete.
            tarfile.TarError: If archive extraction fails.

        Notes:
            This mirrors stage 1 of the ESPnet2 ``local/data.sh`` recipe:
            1. Return early when the corpus is already reachable.
            2. Download ``archive_name`` from the first URL that responds.
            3. Unpack the nested per-split archives and rename them.
            4. Verify that every required split is present.

        Examples:
            >>> builder = MELDBuilder()
            >>> builder.prepare_source("egs3/meld/cls")
        """
        recipe_root = Path(recipe_dir).resolve()
        if self.is_source_prepared(recipe_dir=recipe_root, source_dir=source_dir):
            return

        source_root = resolve_download_root(recipe_root, source_dir=source_dir)
        source_root.mkdir(parents=True, exist_ok=True)

        archive = source_root / str(_CFG["archive_name"])
        if not archive.is_file():
            _download_archive(archive)

        _unpack_archive(source_root, archive)

        missing = missing_source_entries(source_root)
        if missing:
            raise RuntimeError(
                "MELD source is incomplete after unpacking. Missing:\n"
                + "\n".join(f"  - {path}" for path in missing)
            )
        logger.info("MELD source prepared under %s", source_root)

    def is_built(
        self,
        recipe_dir: str | Path,
        **_kwargs,
    ) -> bool:
        """Check whether every split manifest already exists."""
        data_root = resolve_data_root(Path(recipe_dir).resolve())
        return all(
            (data_root / relpath).is_file()
            for relpath in _CFG["manifest_paths"].values()
        )

    def build(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> None:
        """Convert clips to WAV and write one TSV manifest per split.

        Args:
            recipe_dir: Recipe root directory.
            source_dir: Optional explicit MELD source root.
            **_kwargs: Unused extra options for API compatibility.

        Raises:
            FileNotFoundError: If the MELD source tree cannot be resolved.
            RuntimeError: If ``ffmpeg`` is unavailable or a split yields no
                usable utterance.
            subprocess.CalledProcessError: If audio conversion fails.

        Notes:
            Build flow:
            1. Resolve the source root and the output root.
            2. Read each split CSV and drop excluded utterance ids.
            3. Convert each clip to mono WAV at ``sampling_rate``.
            4. Write one TSV manifest per split.

            The label inventory is not written here: the ``prepare_labels``
            stage derives it from the training manifest.
        """
        recipe_root = Path(recipe_dir).resolve()
        source_root = resolve_source_root(recipe_root, source_dir=source_dir)
        data_root = resolve_data_root(recipe_root)

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg not found in PATH")

        excluded = {str(utt_id) for utt_id in _CFG["excluded_utterance_ids"]}
        sampling_rate = str(_CFG["sampling_rate"])

        for split, spec in _CFG["splits"].items():
            annotation = source_root / _CFG["metadata_subdir"] / spec["csv_name"]
            clip_dir = source_root / _CFG["audio_subdir"] / spec["audio_subdir"]
            wav_dir = data_root / _CFG["wav_subdir"] / split
            wav_dir.mkdir(parents=True, exist_ok=True)

            entries: list[tuple[str, Path, str]] = []
            skipped = 0
            with annotation.open("r", encoding="utf-8", errors="replace") as fh:
                for row in csv.DictReader(fh):
                    utt_id = _utterance_id(row, split)
                    if utt_id in excluded:
                        continue

                    label = row["Emotion"].strip()
                    clip = (
                        clip_dir
                        / f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
                    )
                    if not clip.is_file():
                        skipped += 1
                        continue

                    wav = (wav_dir / f"{utt_id}.wav").resolve()
                    if not wav.exists():
                        subprocess.run(
                            [
                                ffmpeg,
                                "-i",
                                str(clip),
                                "-ac",
                                "1",
                                "-ar",
                                sampling_rate,
                                "-f",
                                "wav",
                                "-vn",
                                "-y",
                                "-hide_banner",
                                "-loglevel",
                                "error",
                                str(wav),
                            ],
                            check=True,
                        )
                    entries.append((utt_id, wav, label))

            if not entries:
                raise RuntimeError(f"No usable utterance found for split: {split}")
            if skipped:
                logger.warning(
                    "%s: skipped %d utterance(s) without audio under %s",
                    split,
                    skipped,
                    clip_dir,
                )

            manifest = data_root / _CFG["manifest_paths"][split]
            manifest.parent.mkdir(parents=True, exist_ok=True)
            # Write to a part file and rename, so an interrupted build cannot
            # leave a truncated manifest that `is_built` would accept.
            part = manifest.with_suffix(manifest.suffix + ".part")
            with part.open("w", encoding="utf-8") as fh:
                for utt_id, wav, label in sorted(entries):
                    fh.write(f"{utt_id}\t{wav}\t{label}\n")
            part.replace(manifest)
            logger.info("%s: wrote %d entries to %s", split, len(entries), manifest)
