"""VCTK-Noisy dataset builder.

The public VCTK-DEMAND release already ships wav directories, so this builder
only validates that layout. Download and extract the corpus yourself, then
point ``dataset_dir`` at the root that contains the four wav folders.
"""

from __future__ import annotations

from pathlib import Path

from espnet3.components.data.dataset_builder import DatasetBuilder

_REQUIRED_DIRS = (
    "clean_trainset_28spk_wav",
    "noisy_trainset_28spk_wav",
    "clean_testset_wav",
    "noisy_testset_wav",
)


class VCTKNoisyBuilder(DatasetBuilder):
    """Validate the on-disk VCTK-DEMAND layout used by ``VCTKNoisyDataset``."""

    def _root(self, dataset_dir: str | Path | None, **_kwargs) -> Path:
        if dataset_dir is None:
            raise ValueError(
                "dataset_dir is required. Set it in conf/training.yaml to your "
                "VCTK-DEMAND root (the directory that contains clean_/noisy_ "
                "train and test wav folders)."
            )
        return Path(dataset_dir).expanduser().resolve()

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
