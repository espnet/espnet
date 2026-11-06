"""Dataset builder for the mini_an4 kNN-VC recipe.

The corpus ships with the repository as ``downloads.tar.gz`` (a symlink to the
one in ``egs3/mini_an4/asr``), so ``create_dataset`` only has to extract it.
Nothing is downloaded.
"""

from __future__ import annotations

import shutil
import tarfile
from importlib import resources
from pathlib import Path

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults


def _load_builder_config() -> dict:
    """Load the ``builder`` section of this dataset module's ``config.yaml``."""
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()


def speaker_root(recipe_dir: Path | str | None = None) -> Path:
    """Return the directory holding one subdirectory per speaker.

    Args:
        recipe_dir: Recipe root; defaults to this recipe's directory.

    Returns:
        Path to ``downloads/an4/wav/an4_clstk``.
    """
    root = Path(recipe_dir) if recipe_dir else Path(__file__).resolve().parents[1]
    return root / _CFG["dataset_path"] / _CFG["sph_subdir"]


class MiniAn4Builder(DatasetBuilder):
    """Extract the bundled an4 archive into ``downloads/``."""

    def __init__(self, recipe_dir: Path | str | None = None, **_kwargs) -> None:
        """Record the recipe root the archive is extracted under."""
        self.recipe_dir = (
            Path(recipe_dir) if recipe_dir else Path(__file__).resolve().parents[1]
        )

    def is_source_prepared(self, **_kwargs) -> bool:
        """Return whether the archive has already been extracted."""
        root = speaker_root(self.recipe_dir)
        return root.is_dir() and any(root.glob("*/*.sph"))

    def prepare_source(self, **_kwargs) -> None:
        """Extract ``downloads.tar.gz`` unless it is already extracted.

        Raises:
            FileNotFoundError: If the bundled archive is missing.
        """
        if self.is_source_prepared():
            return
        archive = (self.recipe_dir / _CFG["archive_path"]).resolve()
        if not archive.is_file():
            raise FileNotFoundError(f"Bundled corpus archive not found: {archive}")
        target = self.recipe_dir / _CFG["dataset_path"]
        staging = target.with_name(target.name + ".tmp")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        with tarfile.open(archive) as tar:
            tar.extractall(staging)
        # The archive expands as downloads/..., so lift one level when needed.
        nested = staging / _CFG["dataset_path"]
        source = nested if nested.is_dir() else staging
        shutil.rmtree(target, ignore_errors=True)
        shutil.move(str(source), str(target))
        shutil.rmtree(staging, ignore_errors=True)

    def is_built(self, **_kwargs) -> bool:
        """Return whether the dataset is ready; extraction is all it needs."""
        return self.is_source_prepared()

    def build(self, **_kwargs) -> None:
        """Extract the corpus if it is not already in place."""
        self.prepare_source()
