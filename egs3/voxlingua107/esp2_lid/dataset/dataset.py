"""VoxLingua107 dataset backed by generated TSV manifests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset as TorchDataset

from egs3.voxlingua107.esp2_lid.dataset.builder import (
    _BUILDING,
    resolve_metadata_root,
)
from espnet3.utils.config_utils import load_config_with_defaults

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
_KNOWN_SPLITS = {str(split) for split in _CONFIG["dataset"]["supported_splits"]}


@dataclass(frozen=True)
class VoxLingua107Example:
    """One manifest entry."""

    utt_id: str
    audio_path: Path
    language: str


def _read_manifest(path: Path) -> list[VoxLingua107Example]:
    examples = []
    with path.open("r", encoding="utf-8") as manifest:
        for raw_line in manifest:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            utt_id, audio_path, language = line.split("\t")
            examples.append(VoxLingua107Example(utt_id, Path(audio_path), language))
    if not examples:
        raise RuntimeError(f"Manifest is empty: {path}")
    return examples


class VoxLingua107Dataset(TorchDataset):
    """Dataset returning raw speech and a language label."""

    def __init__(
        self,
        split: str,
        sample_rate: int = 16000,
        recipe_dir: str | Path | None = None,
        data_dir: str | Path | None = None,
        speed_perturb_factors: Sequence[float] | None = None,
    ) -> None:
        """Read only the requested split's manifest and referenced audio.

        Audio paths come from the manifest. Other splits and their source audio
        are not required.
        ``speed_perturb_factors`` expands train into one copy per factor. Audio
        is perturbed on read, preserving labels; dev always uses original audio.

        Args:
            split: Prepared train or dev split.
            sample_rate: Expected waveform sample rate, normally 16000.
            recipe_dir: Recipe root used to locate prepared manifests.
            data_dir: Explicit metadata root overriding the recipe default.
            speed_perturb_factors: Training speed factors; default is [1.0].
        """
        self.split = str(split)
        if self.split not in _KNOWN_SPLITS:
            known = ", ".join(sorted(_KNOWN_SPLITS))
            raise ValueError(f"Unknown split '{self.split}'. Expected one of: {known}")

        metadata_root = resolve_metadata_root(recipe_dir, data_dir)
        manifest = metadata_root / self.split / "manifest.tsv"
        if (metadata_root / _BUILDING).exists() or not manifest.is_file():
            raise FileNotFoundError(
                f"VoxLingua107 manifest is missing or incomplete: {manifest}. "
                "Run the create_dataset stage first."
            )

        self.examples = _read_manifest(manifest)
        self.sample_rate = int(sample_rate)
        self.speed_perturb_factors = (
            tuple(float(factor) for factor in (speed_perturb_factors or (1.0,)))
            if self.split == "train"
            else (1.0,)
        )

    def __len__(self) -> int:
        """Return the number of samples including configured speed variants."""
        return len(self.examples) * len(self.speed_perturb_factors)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Read one waveform and its language code without copying source audio."""
        factor_idx, example_idx = divmod(int(idx), len(self.examples))
        factor = self.speed_perturb_factors[factor_idx]
        example = self.examples[example_idx]
        speech, sample_rate = sf.read(example.audio_path, dtype="float32")
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"Expected {self.sample_rate} Hz, got {sample_rate} Hz: "
                f"{example.audio_path}"
            )
        if speech.ndim == 2:
            speech = speech.mean(axis=1)
        if factor != 1.0:
            import torch

            from espnet2.layers.augmentation import speed_perturb

            speech = speed_perturb(
                torch.from_numpy(speech), sample_rate, factor
            ).numpy()
        return {
            "speech": np.asarray(speech, dtype=np.float32),
            "lid_labels": example.language,
        }
