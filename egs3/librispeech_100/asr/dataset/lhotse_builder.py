"""LibriSpeech 100h dataset builder using Lhotse."""

from __future__ import annotations

import os
import subprocess
from importlib import resources
from pathlib import Path
from typing import Iterable

from lhotse import CutSet, RecordingSet, SupervisionSet
from lhotse.recipes import prepare_librispeech

from egs3.librispeech_100.asr.dataset.data_utils import LhotseElement
from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("lhotse_config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
_REQUIRED_SPLITS = {str(split) for split in _CFG["required_splits"]}


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
        + f"Place the corpus under <recipe_dir>/{_CFG['dataset_path']}/LibriSpeech "
        + f"or set {env_var} to the dataset root."
    )


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


def missing_required_splits(source_root: Path) -> list[str]:
    """Return required split names that are missing from ``source_root``."""
    return [
        str(split)
        for split in _CFG["required_splits"]
        if not (source_root / str(split)).is_dir()
    ]


class LibriSpeech100LhotseBuilder(DatasetBuilder):
    """
    This recipe relies on lhotse data preparation and provides cutsets
    to the dataset class. If the path to the manifests is empty,
     it will first create the manifests.
    """

    def __init__(self, file_format: str = "jsonl.gz"):

        self.file_format = file_format

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
        """Validate that the LibriSpeech source tree is already available.

        Args:
            recipe_dir: Recipe root directory.
            source_dir: Optional override pointing to a LibriSpeech parent/root.
            **_kwargs: Unused extra options for API compatibility.

        Raises:
            FileNotFoundError: If the LibriSpeech root or required splits are
                missing.
        """
        recipe_root = Path(recipe_dir).resolve()
        source_root = resolve_source_root(recipe_root, source_dir=source_dir)
        missing = missing_required_splits(source_root)

        if missing:
            raise FileNotFoundError(
                "LibriSpeech source is incomplete. Missing split directories: "
                + ", ".join(missing)
            )

    def _is_manifest_source_ready(self, dataset_dir: str | Path):
        assert self.file_format is not None, "First call the build() function."
        dataset_dir = (
            Path(dataset_dir)
            if dataset_dir is not None and Path(dataset_dir).is_dir()
            else Path(_CFG["dataset_dir"])
        )

        return all(
            (dataset_dir / f"{split}_recordings.{self.file_format}").is_file()
            and (dataset_dir / f"{split}_supervisions.{self.file_format}").is_file()
            for split in _CFG["required_splits"]
        )

    @staticmethod
    def _write_manifests(
        recipe_dir: str | Path,
        dataset_dir: str | Path,
        source_dir: str | Path | None = None,
        num_jobs: int = 8,
        file_format: str = "jsonl.gz",
    ) -> None:
        """Write Lhotse recording and supervision manifests."""
        recipe_root = Path(recipe_dir).resolve()
        source_root = resolve_source_root(recipe_root, source_dir=source_dir)
        dataset_dir = Path(dataset_dir)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        manifests = prepare_librispeech(
            corpus_dir=source_root,
            dataset_parts=_REQUIRED_SPLITS,
            num_jobs=num_jobs,
        )

        for split, split_manifests in manifests.items():
            for element in LhotseElement:
                split_manifests[element.value].to_file(
                    dataset_dir / f"{split}_{element.value}.{file_format}"
                )

    def load_cutsets(
        self,
        split: str,
        dataset_dir: str | Path | None = None,
    ) -> CutSet:
        """Load CutSets from recording and supervision manifests for a split."""

        if not self._is_manifest_source_ready(dataset_dir):
            raise RuntimeError(
                "Manifest files are not available. Call build() first to generate "
                "the manifests, then call load_cutsets()."
            )

        if dataset_dir is None or not Path(dataset_dir).is_dir():
            dataset_dir = Path(_CFG["dataset_dir"])
        else:
            dataset_dir = Path(dataset_dir).resolve()

        recordings = RecordingSet.from_file(
            dataset_dir
            / f"{split}_{LhotseElement.RecordingSet.value}.{self.file_format}"
        )
        supervisions = SupervisionSet.from_file(
            dataset_dir
            / f"{split}_{LhotseElement.SupervisionSet.value}.{self.file_format}"
        )

        return CutSet.from_manifests(
            recordings=recordings,
            supervisions=supervisions,
        )

    def is_built(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        dataset_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Return source readiness because this recipe has no build artifacts."""
        dataset_dir = dataset_dir or _CFG["dataset_dir"]

        return self.is_source_prepared(
            recipe_dir=recipe_dir,
            source_dir=source_dir,
        ) and self._is_manifest_source_ready(dataset_dir)

    def build(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        dataset_dir: str | Path | None = None,
        **_kwargs,
    ) -> None:
        """No-op build step for raw-directory-backed LibriSpeech access."""
        dataset_dir = (
            Path(dataset_dir)
            if dataset_dir is not None and Path(dataset_dir).is_dir()
            else Path(_CFG["dataset_dir"])
        )
        self.prepare_source(recipe_dir=recipe_dir, source_dir=source_dir)

        if not self._is_manifest_source_ready(dataset_dir):
            self._write_manifests(
                recipe_dir=recipe_dir,
                dataset_dir=dataset_dir,
                source_dir=source_dir,
                file_format=self.file_format,
            )
