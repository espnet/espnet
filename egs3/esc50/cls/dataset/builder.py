"""ESC-50 dataset builder."""

from __future__ import annotations

import csv
import logging
import os
from importlib import resources
from pathlib import Path
from typing import Iterator

from omegaconf import OmegaConf

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.systems.cls.audio_conversion_provider import AudioConversionProvider
from espnet3.systems.cls.audio_conversion_runner import AudioConversionRunner
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()


def _iter_source_candidates(
    recipe_root: Path,
    source_dir: str | Path | None = None,
) -> Iterator[Path]:
    """Yield candidate ESC-50 source roots in priority order."""
    if source_dir is not None:
        yield Path(source_dir).expanduser()

    env_path = os.environ.get(str(_CFG["source_env_var"]))
    if env_path:
        yield Path(env_path).expanduser()

    yield recipe_root / _CFG["dataset_path"]


def _missing_source_entries(source_root: Path) -> list[str]:
    """Return required source paths that are absent from ``source_root``."""
    missing: list[str] = []

    metadata = source_root / _CFG["metadata_path"]
    if not metadata.is_file():
        missing.append(str(metadata))

    audio_dir = source_root / _CFG["audio_subdir"]
    if not audio_dir.is_dir():
        missing.append(str(audio_dir))

    return missing


def _resolve_source_root(
    recipe_root: Path,
    source_dir: str | Path | None = None,
) -> Path:
    """Resolve the usable ESC-50 source root for this recipe.

    Args:
        recipe_root: Recipe root directory.
        source_dir: Optional explicit source root that wins over the
            environment variable and the in-recipe download directory.

    Returns:
        Path to a directory holding both ``meta/esc50.csv`` and ``audio/``.

    Raises:
        FileNotFoundError: If no candidate directory holds a complete corpus.
    """
    checked: list[str] = []
    for candidate in _iter_source_candidates(recipe_root, source_dir):
        checked.append(str(candidate))
        if not _missing_source_entries(candidate):
            return candidate

    env_var = str(_CFG["source_env_var"])
    raise FileNotFoundError(
        "ESC-50 source not found. Checked these locations:\n"
        + "\n".join(f"  - {path}" for path in checked)
        + "\n"
        + "Download the corpus from https://github.com/karolpiczak/ESC-50, "
        + f"place it under <recipe_dir>/{_CFG['dataset_path']} or set "
        + f"{env_var} to the dataset root. The directory must contain "
        + f"{_CFG['metadata_path']} and {_CFG['audio_subdir']}/."
    )


def _resolve_fold() -> str:
    """Resolve the held-out fold, which serves as both valid and test."""
    return str(os.environ.get(str(_CFG["fold_env_var"])) or _CFG["default_fold"])


def resolve_output_root(recipe_root: Path) -> Path:
    """Resolve the root every fold shares, holding the converted audio."""
    env_var = _CFG.get("output_env_var")
    if env_var:
        env_path = os.environ.get(str(env_var))
        if env_path:
            return Path(env_path).expanduser()
    return recipe_root / _CFG["data_path"]


def resolve_data_root(recipe_root: Path) -> Path:
    """Resolve this fold's manifest root, matching ``data_dir`` in the config."""
    return resolve_output_root(recipe_root) / f"fold{_resolve_fold()}"


def _read_metadata(metadata_path: Path) -> list[dict]:
    """Read ``meta/esc50.csv`` as rows, sorted by file name.

    The split below is drawn over row positions, so sorting pins down which
    clip lands where. The distributed metadata is already in this order, so
    the result matches the ESPnet2 recipe.
    """
    with metadata_path.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        raise RuntimeError(f"No rows found in metadata: {metadata_path}")

    return sorted(rows, key=lambda row: row["filename"])


def _split_rows(rows: list[dict]) -> dict[str, list[dict]]:
    """Partition metadata rows into train, valid, and test.

    ESPnet2 trains on the four folds it keeps and uses the held-out fold for
    both validation and scoring, so ``valid`` and ``test`` are the same rows.

    Args:
        rows: Every row of ``meta/esc50.csv``.

    Returns:
        A mapping from split name to the rows belonging to it.

    Raises:
        RuntimeError: If the configured fold holds no clip.
    """
    fold = _resolve_fold()
    held_out = [row for row in rows if row["fold"] == fold]
    if not held_out:
        folds = sorted({row["fold"] for row in rows})
        raise RuntimeError(
            f"No clip found for fold={fold}. Available folds: {', '.join(folds)}"
        )

    return {
        "train": [row for row in rows if row["fold"] != fold],
        "valid": held_out,
        "test": held_out,
    }


def _convert_clips(conversions: list[tuple[Path, Path]], data_root: Path) -> None:
    """Resample every clip to 16 kHz mono WAV, fanning the work out.

    ESC-50 ships 44.1 kHz audio while ``BeatsEncoder`` needs 16 kHz. The source
    tree is often read-only, so the output goes under the recipe's data root.

    Args:
        conversions: ``(source clip, destination WAV)`` pairs.
        data_root: Output root, used to place the shard bookkeeping.

    Raises:
        RuntimeError: If ``ffmpeg`` is not installed.
        subprocess.CalledProcessError: If a conversion fails.
    """
    if not conversions:
        return

    provider = AudioConversionProvider(
        config=OmegaConf.create({}),
        params={
            "jobs": conversions,
            "sampling_rate": _CFG["sampling_rate"],
        },
    )
    # resume=False: a shard marked done by an earlier run says nothing about
    # whether its WAV files are still on disk. The per-file existence check in
    # the runner is what makes a rerun cheap.
    runner = AudioConversionRunner(
        provider=provider,
        output_dir=data_root / "conversion_shards",
        resume=False,
    )
    results = runner(list(range(len(conversions))))
    converted = sum(1 for record in results if record["converted"])
    logger.info(
        "Converted %d clip(s); %d already present",
        converted,
        len(results) - converted,
    )


class ESC50Builder(DatasetBuilder):
    """Prepare and build ESC-50 assets for ESPnet3 recipes."""

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **kwargs,
    ) -> bool:
        """Check whether a complete ESC-50 source tree is reachable."""
        recipe_root = Path(recipe_dir).resolve()
        try:
            _resolve_source_root(recipe_root, source_dir=source_dir)
        except FileNotFoundError:
            return False
        return True

    def prepare_source(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **kwargs,
    ) -> None:
        """Verify that the ESC-50 corpus is reachable.

        This never downloads or writes: ESC-50 is usually mounted read-only, so
        the corpus is validated rather than produced. A failure here is the
        message telling the user where to put it.

        Args:
            recipe_dir: Recipe root directory.
            source_dir: Optional explicit source root that wins over the
                environment variable and the in-recipe download directory.
            **kwargs: Unused extra options for API compatibility.

        Returns:
            None.

        Raises:
            FileNotFoundError: If no candidate location holds the corpus. The
                message lists every path that was probed.
        """
        recipe_root = Path(recipe_dir).resolve()
        source_root = _resolve_source_root(recipe_root, source_dir=source_dir)
        logger.info("ESC-50 source found under %s", source_root)

    def is_built(
        self,
        recipe_dir: str | Path,
        **kwargs,
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
        **kwargs,
    ) -> None:
        """Resample the clips and write one TSV manifest per split.

        Args:
            recipe_dir: Recipe root directory.
            source_dir: Optional explicit ESC-50 source root.
            **kwargs: Unused extra options for API compatibility.

        Returns:
            None.

        Raises:
            FileNotFoundError: If the ESC-50 source tree cannot be resolved.
            RuntimeError: If the metadata is empty, the configured test fold is
                unknown, or a split yields no usable clip.
            subprocess.CalledProcessError: If audio conversion fails.

        Notes:
            The label is the ``category`` column (``dog``, ``washing_machine``).
            Those names carry no whitespace, so each is one ``word`` token. The
            label inventory itself is written by ``prepare_labels``.
        """
        recipe_root = Path(recipe_dir).resolve()
        source_root = _resolve_source_root(recipe_root, source_dir=source_dir)
        data_root = resolve_data_root(recipe_root)

        rows = _read_metadata(source_root / _CFG["metadata_path"])
        audio_dir = source_root / _CFG["audio_subdir"]

        # The converted audio does not depend on the fold, so all five folds
        # share one directory and the clips are resampled once.
        output_root = resolve_output_root(recipe_root)
        wav_dir = output_root / _CFG["wav_subdir"]
        wav_dir.mkdir(parents=True, exist_ok=True)

        split_entries: dict[str, list[tuple[str, Path, str]]] = {}
        conversions: dict[Path, Path] = {}
        for split, split_rows in _split_rows(rows).items():
            entries: list[tuple[str, Path, str]] = []
            missing: list[str] = []
            for row in split_rows:
                clip = audio_dir / row["filename"]
                if not clip.is_file():
                    missing.append(str(clip))
                    continue

                utt_id = Path(row["filename"]).stem
                wav = (wav_dir / f"{utt_id}.wav").resolve()
                conversions[wav] = clip
                entries.append((utt_id, wav, row["category"].strip()))

            # ESC-50 ships one file per metadata row, so a missing clip means
            # an incomplete copy. Skipping would quietly shrink the split.
            if missing:
                raise RuntimeError(
                    f"{len(missing)} clip(s) listed in the metadata are missing "
                    f"from {audio_dir} for split '{split}':\n"
                    + "\n".join(f"  - {path}" for path in missing[:10])
                    + ("\n  ..." if len(missing) > 10 else "")
                )
            if not entries:
                raise RuntimeError(f"No usable clip found for split: {split}")
            split_entries[split] = entries

        _convert_clips([(src, dst) for dst, src in conversions.items()], output_root)

        for split, entries in split_entries.items():
            manifest = data_root / _CFG["manifest_paths"][split]
            manifest.parent.mkdir(parents=True, exist_ok=True)
            # Write to a part file and rename, so an interrupted build cannot
            # leave a truncated manifest that `is_built` would accept.
            part = manifest.with_suffix(manifest.suffix + ".part")
            with part.open("w", encoding="utf-8") as fh:
                for utt_id, wav, label in sorted(entries, key=lambda row: row[0]):
                    fh.write(f"{utt_id}\t{wav}\t{label}\n")
            part.replace(manifest)
            logger.info("%s: wrote %d entries to %s", split, len(entries), manifest)
