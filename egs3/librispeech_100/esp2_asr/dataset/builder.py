"""LibriSpeech 100h dataset builder."""

from __future__ import annotations

import logging
import os
import shutil
import tarfile
from importlib import resources
from pathlib import Path
from typing import Iterable

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.download_utils import download_url, extract_targz

logger = logging.getLogger(__name__)


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()


def resolve_librispeech_root(data_dir: str | Path) -> Path:
    """Resolve a path to the on-disk ``LibriSpeech`` root."""
    candidate = Path(data_dir)
    if (candidate / "LibriSpeech").is_dir():
        return candidate / "LibriSpeech"
    if candidate.name == "LibriSpeech" and candidate.is_dir():
        return candidate
    raise FileNotFoundError(
        "Could not find LibriSpeech root. Expected either:\n"
        f"  - {candidate}/LibriSpeech/\n"
        f"  - {candidate} (when it is the LibriSpeech directory itself)"
    )


def iter_source_candidates(
    recipe_root: Path,
    source_dir: str | Path | None,
) -> Iterable[Path]:
    """Yield candidate directories that may contain LibriSpeech."""
    yield recipe_root / _CFG["dataset_path"]

    if source_dir is not None:
        yield Path(source_dir)

    env_var = str(_CFG["source_env_var"])
    env_path = os.environ.get(env_var)
    if env_path:
        yield Path(env_path)


def resolve_source_root(
    recipe_root: Path,
    source_dir: str | Path | None = None,
) -> Path:
    """Resolve the usable LibriSpeech source root for this recipe."""
    checked: list[str] = []
    for candidate in iter_source_candidates(recipe_root, source_dir):
        checked.append(str(candidate))
        try:
            return resolve_librispeech_root(candidate)
        except FileNotFoundError:
            continue

    env_var = str(_CFG["source_env_var"])
    raise FileNotFoundError(
        "LibriSpeech source not found. Checked these locations:\n"
        + "\n".join(f"  - {path}" for path in checked)
        + "\n"
        + "Run the create_dataset stage to download it into "
        + f"<recipe_dir>/{_CFG['dataset_path']}, or set {env_var} to an "
        + "existing LibriSpeech root."
    )


def missing_required_splits(source_root: Path) -> list[str]:
    """Return required split names that are missing from ``source_root``."""
    return [
        str(split)
        for split in _CFG["required_splits"]
        if not (source_root / str(split)).is_dir()
    ]


def download_and_extract_if_needed(split: str, dataset_dir: Path) -> None:
    """Download and extract one OpenSLR split into ``dataset_dir/LibriSpeech``.

    An already extracted split is skipped and an already downloaded archive is
    reused. The archive is downloaded to a ``.part`` file and extracted into a
    staging directory first, so an interrupted run never looks complete.

    Args:
        split: LibriSpeech split name such as ``dev-clean``.
        dataset_dir: Directory holding the archives and the ``LibriSpeech`` tree.

    Raises:
        RuntimeError: If the download or the extraction fails.
    """
    librispeech_root = dataset_dir / "LibriSpeech"
    extracted_dir = librispeech_root / split
    if extracted_dir.is_dir():
        logger.info("Skip split '%s' (already exists): %s", split, extracted_dir)
        return

    dataset_dir.mkdir(parents=True, exist_ok=True)
    archive_name = f"{split}.tar.gz"
    archive_path = dataset_dir / archive_name

    if archive_path.is_file():
        logger.info("Archive exists, skip download: %s", archive_path)
    else:
        url = f"{str(_CFG['download_url']).rstrip('/')}/{archive_name}"
        partial_path = dataset_dir / f"{archive_name}.part"
        try:
            download_url(url=url, dst_path=partial_path, logger=logger)
        except (OSError, ValueError) as e:
            partial_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to download {url}: {e}\n"
                "Check network access and builder.download_url in "
                "dataset/config.yaml, or place an existing corpus at "
                f"{librispeech_root} or set {_CFG['source_env_var']}."
            ) from e
        partial_path.replace(archive_path)

    staging_dir = dataset_dir / f".{split}.extracting"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    try:
        extract_targz(archive_path, staging_dir, logger)
        staged_root = staging_dir / "LibriSpeech"
        if not (staged_root / split).is_dir():
            raise tarfile.TarError(f"no LibriSpeech/{split}/ directory inside")
        librispeech_root.mkdir(parents=True, exist_ok=True)
        # Every archive also carries the shared top-level metadata files
        # (BOOKS.TXT, SPEAKERS.TXT, ...); keep the first copy.
        for item in staged_root.iterdir():
            if not (librispeech_root / item.name).exists():
                item.rename(librispeech_root / item.name)
    except (OSError, EOFError, tarfile.TarError) as e:
        raise RuntimeError(
            f"Failed to extract {archive_path}: {e}\n"
            "The archive may be incomplete or corrupt. Delete it and rerun "
            "create_dataset to download it again."
        ) from e
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    logger.info("Finished split: %s", split)


class LibriSpeech100Builder(DatasetBuilder):
    """Download or locate the LibriSpeech source tree for the recipe.

    This recipe reads the original LibriSpeech directory layout directly during
    training and inference, so there are no build artifacts. Sources are looked
    up in this order, and the first location holding a ``LibriSpeech`` root is
    used:

    1. ``<recipe_dir>/<builder.dataset_path>`` (``download`` by default)
    2. ``source_dir``
    3. the ``LIBRISPEECH`` environment variable

    Each location may be the ``LibriSpeech`` directory itself or its parent.
    When no location holds a corpus, or the ``builder.dataset_path`` copy is
    missing splits, ``prepare_source`` downloads the missing splits there.
    Corpora found through ``source_dir`` or ``LIBRISPEECH`` are never modified.
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Check whether the required LibriSpeech splits are available."""
        recipe_root = Path(recipe_dir).resolve()
        try:
            source_root = resolve_source_root(recipe_root, source_dir=source_dir)
        except FileNotFoundError:
            return False
        return not missing_required_splits(source_root)

    def prepare_source(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> None:
        """Reuse an existing LibriSpeech tree or download the missing splits.

        Args:
            recipe_dir: Recipe root directory.
            source_dir: Optional override pointing to a LibriSpeech parent/root.
            **_kwargs: Unused extra options for API compatibility.

        Raises:
            FileNotFoundError: If a corpus found through ``source_dir`` or
                ``LIBRISPEECH`` is missing required splits.
            RuntimeError: If a download or extraction fails.
        """
        recipe_root = Path(recipe_dir).resolve()
        download_dir = recipe_root / _CFG["dataset_path"]
        try:
            source_root = resolve_source_root(recipe_root, source_dir=source_dir)
        except FileNotFoundError:
            source_root = download_dir / "LibriSpeech"

        missing = missing_required_splits(source_root)
        if not missing:
            return
        if source_root != download_dir / "LibriSpeech":
            raise FileNotFoundError(
                f"LibriSpeech source at {source_root} is missing split "
                f"directories: {', '.join(missing)}. Add them there, or unset "
                f"source_dir/{_CFG['source_env_var']} to download the corpus "
                f"into {download_dir}."
            )

        for split in missing:
            download_and_extract_if_needed(split, download_dir)

        missing = missing_required_splits(source_root)
        if missing:
            raise FileNotFoundError(
                f"LibriSpeech split directories still missing after download: "
                f"{', '.join(missing)} under {source_root}."
            )

    def is_built(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Return source readiness because this recipe has no build artifacts."""
        return self.is_source_prepared(
            recipe_dir=recipe_dir,
            source_dir=source_dir,
        )

    def build(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> None:
        """No-op build step for raw-directory-backed LibriSpeech access.

        Raises:
            FileNotFoundError: If the source has not been prepared.
        """
        if not self.is_source_prepared(recipe_dir=recipe_dir, source_dir=source_dir):
            raise FileNotFoundError(
                "LibriSpeech source is not prepared. Call prepare_source first."
            )
