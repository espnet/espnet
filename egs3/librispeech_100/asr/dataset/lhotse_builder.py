"""LibriSpeech 100h dataset builder using Lhotse."""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path
from typing import Iterable

from lhotse import CutSet, RecordingSet, SupervisionSet
from lhotse.recipes import (
    prepare_librispeech
)

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults

from egs3.librispeech_100.dataset.utils import LhotseElement


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("lhotse_config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()
_REQUIRED_SPLITS = {str(split) for split in _CFG["required_splits"]}


class LibriSpeech100LhotseBuilder(DatasetBuilder):
    """

    This recipe relies on lhotse data preparation and provides cutsets to the dataset class
    If the path to the manifests are empty, it will first create the manifests.
    """

    def _is_source_prepared(self, manifest_dir: str | Path, **_kwargs) -> bool:
        """Check whether the manifest files are already available.

        Args:
            manifest_dir: manifest directory.
            **_kwargs: Unused extra options for API compatibility.

        Returns:
            ``True`` if ``<manifest_dir>/{data_split}_{supervisions,recordings}.json.gz``
            for all required splits exist; otherwise ``False``.
        """
        recipe_root = Path(recipe_dir).resolve()
        self.manifest_dir = recipe_root / _CFG["manifest_dir"]

        return (
                manifest_dir.is_dir()
                and all(
            (manifest_dir / f"{split}_{element.value}.json.gz").is_file()
            for split in _REQUIRED_SPLITS
            for element in LhotseElement
        )
        )

    from pathlib import Path
    from typing import Mapping

    @staticmethod
    def _write_manifests(
            dataset_path: str | Path,
            manifest_dir: str | Path,
            num_jobs: int = 8,
    ) -> None:
        """Write Lhotse recording and supervision manifests."""
        dataset_dir = Path(dataset_dir)
        manifest_dir = Path(manifest_dir)
        manifest_dir.mkdir(parents=True, exist_ok=True)

        manifests = prepare_librispeech(
            corpus_dir=dataset_path,
            dataset_parts=_REQUIRED_SPLITS,
            num_jobs=num_jobs,
        )

        for split, split_manifests in manifests.items():
            for element in LhotseElement:
                split_manifests[element.value].to_file(
                    manifest_dir / f"{split}_{element.value}.{file_format}"
                )

    @staticmethod
    def load_cutsets(
            manifest_dir: str | Path,
            split: str,
    ) -> CutSet:
        """Load CutSets from recording and supervision manifests for a split."""

        recordings = RecordingSet.from_file(
            manifest_dir / f"{split}_{LhotseElement.RecordingSet.value}.json.gz"
        )
        supervisions = SupervisionSet.from_file(
            manifest_dir / f"{split}_{LhotseElement.SupervisionSet.value}.json.gz"
        )

        cuts[str(split)] = CutSet.from_manifests(
            recordings=recordings,
            supervisions=supervisions,
        )

        return cuts



    def build(
        self,
        dataset_path: str | Path,
        manifest_dir: str | Path,
        **_kwargs,
    ) -> None:
        """No-op build step for raw-directory-backed LibriSpeech access."""
        if not self._is_source_prepared(manifest_dir):
            self._write_manifests(dataset_path=dataset_path, manifest_dir=manifest_dir)
