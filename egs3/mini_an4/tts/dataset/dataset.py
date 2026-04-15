"""Mini AN4 TTS dataset implementation backed by TSV manifests."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset as TorchDataset

from egs3.mini_an4.tts.dataset.builder import MiniAn4TTSBuilder
from espnet3.utils.config_utils import load_config_with_defaults

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
_DATASET_CFG = _CONFIG["dataset"]
_BUILDER_CFG = _CONFIG["builder"]

_SPLIT_MANIFEST_PATHS: dict[str, str] = {
    str(split): str(relpath)
    for split, relpath in _DATASET_CFG["split_manifest_paths"].items()
}


@dataclass(frozen=True)
class ManifestEntry:
    """One manifest row for TTS: utterance id, wav path, and transcript text."""

    utt_id: str
    wav_path: Path
    text: str


def _read_manifest(manifest_path: Path) -> list[ManifestEntry]:
    """Read ``utt_id<TAB>wav_path<TAB>text`` lines as manifest entries."""
    entries: list[ManifestEntry] = []

    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            utt_id, wav_path, text = line.split("\t", maxsplit=2)
            entries.append(
                ManifestEntry(
                    utt_id=utt_id,
                    wav_path=Path(wav_path),
                    text=text,
                )
            )

    if not entries:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")

    return entries


class MiniAn4TTSDataset(TorchDataset):
    """Mini AN4 TTS dataset that returns speech/text samples.

    This dataset uses the recipe-local ``dataset`` module convention introduced
    in ESPnet3. The configured ``split`` is resolved to a manifest path under
    ``data/manifest`` and, if needed, the associated builder prepares the raw
    Mini AN4 source tree and task-ready manifests before samples are loaded.

    Args:
        split: Dataset split name. Supported values are ``"train"``,
            ``"valid"``, and ``"test"``.
        recipe_dir: Recipe root directory. When omitted, the current recipe
            directory is inferred from this module location.
        include_utt_id: Whether to include ``utt_id`` in returned samples.
            This is useful for inference so generated artifacts keep the source
            utterance ids.

    Returns:
        A Torch dataset whose items are dicts containing:
        - ``speech``: float32 waveform array
        - ``text``: transcript string
        - ``utt_id``: included only when ``include_utt_id=True``

    Raises:
        ValueError: If ``split`` is unsupported.
        FileNotFoundError: If the resolved manifest file does not exist.
        RuntimeError: If the manifest is empty after dataset preparation.

    Examples:
        >>> dataset = MiniAn4TTSDataset(split="train", recipe_dir="egs3/mini_an4/tts")
        >>> sample = dataset[0]
        >>> sorted(sample.keys())
        ['speech', 'text']

        >>> test_set = MiniAn4TTSDataset(
        ...     split="test",
        ...     recipe_dir="egs3/mini_an4/tts",
        ...     include_utt_id=True,
        ... )
        >>> "utt_id" in test_set[0]
        True
    """

    def __init__(
        self,
        split: str,
        recipe_dir: str | Path | None = None,
        include_utt_id: bool = False,
    ) -> None:
        self.split = split
        self.include_utt_id = include_utt_id
        recipe_root = (
            Path(recipe_dir).resolve()
            if recipe_dir is not None
            else Path(__file__).resolve().parents[1]
        )
        self.dataset_dir = recipe_root / _BUILDER_CFG["data_path"]

        builder = MiniAn4TTSBuilder()
        if not builder.is_source_prepared(recipe_dir=recipe_root):
            builder.prepare_source(recipe_dir=recipe_root)
        if not builder.is_built(recipe_dir=recipe_root):
            builder.build(recipe_dir=recipe_root)

        if split not in _SPLIT_MANIFEST_PATHS:
            known = ", ".join(sorted(_SPLIT_MANIFEST_PATHS))
            raise ValueError(f"Unknown split '{split}'. Expected one of: {known}")

        manifest_path = self.dataset_dir / _SPLIT_MANIFEST_PATHS[split]
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        self._entries = _read_manifest(manifest_path)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self._entries[int(idx)]
        array, _sr = sf.read(str(entry.wav_path))
        sample = {
            "speech": np.asarray(array, dtype=np.float32),
            "text": entry.text,
        }
        if self.include_utt_id:
            sample["utt_id"] = entry.utt_id
        return sample
