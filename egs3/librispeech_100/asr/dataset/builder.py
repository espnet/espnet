"""LibriSpeech 100h dataset builder."""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path
from typing import Iterable

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults


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
        + f"Place the corpus under <recipe_dir>/{_CFG['dataset_path']}/LibriSpeech "
        + f"or set {env_var} to the dataset root."
    )


def missing_required_splits(source_root: Path) -> list[str]:
    """Return required split names that are missing from ``source_root``."""
    return [
        str(split)
        for split in _CFG["required_splits"]
        if not (source_root / str(split)).is_dir()
    ]


class LibriSpeech100Builder(DatasetBuilder):
    """Validate LibriSpeech source availability for the recipe.

    This recipe reads the original LibriSpeech directory layout directly during
    training and inference, so the builder's only responsibility is to ensure
    the expected split directories are available under the configured download
    root (or the ``LIBRISPEECH`` environment variable).
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
        """No-op build step for raw-directory-backed LibriSpeech access."""
        self.prepare_source(recipe_dir=recipe_dir, source_dir=source_dir)
