"""VCTK-Noisy dataset builder.

The public VCTK-DEMAND release already ships wav directories, so this builder
only validates that layout. Download and extract the corpus yourself, then
point ``$VCTK_DEMAND`` (the variable named by ``dataset/config.yaml``) or
``dataset_dir`` in the training config at the root that contains the four wav
folders.

The corpus is not downloaded automatically on purpose. It could be fetched with
``espnet3.utils.download_utils``, but VCTK-DEMAND is large and most users
already keep a shared copy on disk (e.g. under ``/DB``). Downloading it into
each recipe directory would leave redundant copies of the same data in the
user's storage, so the builder reads the existing copy in place instead.
"""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
_REQUIRED_DIRS = tuple(str(name) for name in _CFG["required_dirs"])
_SOURCE_ENV_VAR = str(_CFG["source_env_var"])


def resolve_dataset_dir(dataset_dir: str | Path | None = None) -> Path:
    """Resolve the VCTK-DEMAND root from an explicit path or the environment.

    ``dataset_dir`` wins when a config sets it; otherwise the environment
    variable named by ``builder.source_env_var`` in ``dataset/config.yaml``
    (``VCTK_DEMAND``) is used, so no recipe file has to hard-code a site path.
    """
    if dataset_dir is not None:
        return Path(dataset_dir).expanduser().resolve()
    env_path = os.environ.get(_SOURCE_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser().resolve()
    raise ValueError(
        "VCTK-DEMAND root is not set. Either export "
        f"{_SOURCE_ENV_VAR}=/path/to/vctk_noisy or set dataset_dir in "
        "conf/training.yaml (and conf/inference.yaml). It must point at the "
        "directory that contains: " + ", ".join(_REQUIRED_DIRS) + "."
    )


class VCTKNoisyBuilder(DatasetBuilder):
    """Validate the on-disk VCTK-DEMAND layout used by ``VCTKNoisyDataset``."""

    def _root(self, dataset_dir: str | Path | None, **_kwargs) -> Path:
        return resolve_dataset_dir(dataset_dir)

    def is_source_prepared(
        self,
        recipe_dir: str | Path | None = None,
        dataset_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Return True when the four expected wav directories exist."""
        del recipe_dir
        root = self._root(dataset_dir, **_kwargs)
        return all((root / name).is_dir() for name in _REQUIRED_DIRS)

    def prepare_source(
        self,
        recipe_dir: str | Path | None = None,
        dataset_dir: str | Path | None = None,
        **_kwargs,
    ) -> None:
        """Refuse to download; print a clear path for the user."""
        del recipe_dir
        root = self._root(dataset_dir, **_kwargs)
        missing = [name for name in _REQUIRED_DIRS if not (root / name).is_dir()]
        raise FileNotFoundError(
            "VCTK-DEMAND is not prepared under "
            f"{root}. Missing: {', '.join(missing)}. "
            "Download from https://datashare.ed.ac.uk/handle/10283/2791 "
            "and extract so those directories sit directly under dataset_dir."
        )

    def is_built(
        self,
        recipe_dir: str | Path | None = None,
        dataset_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """No separate manifest build step; source layout is enough."""
        return self.is_source_prepared(
            recipe_dir=recipe_dir, dataset_dir=dataset_dir, **_kwargs
        )

    def build(
        self,
        recipe_dir: str | Path | None = None,
        dataset_dir: str | Path | None = None,
        **_kwargs,
    ) -> None:
        """No-op: ``VCTKNoisyDataset`` reads the wav folders directly."""
        del recipe_dir, dataset_dir, _kwargs
