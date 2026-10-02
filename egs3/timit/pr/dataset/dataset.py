"""TIMIT dataset implementation backed by a TSV manifest."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset as TorchDataset

from egs3.timit.pr.dataset.builder import TimitBuilder, resolve_data_root
from espnet3.utils.config_utils import load_config_with_defaults

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
_DATASET_CFG = _CONFIG["dataset"]


@dataclass(frozen=True)
class ManifestEntry:
    """One manifest row: utterance id, audio path, and IPA reference."""

    utt_id: str
    wav_path: Path
    text: str


def _read_manifest(manifest_path: Path) -> List[ManifestEntry]:
    """Read ``utt_id<TAB>wav_path<TAB>text`` lines as manifest entries."""
    entries: List[ManifestEntry] = []

    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            utt_id, wav_path, text = line.split("\t", maxsplit=2)
            entries.append(
                ManifestEntry(utt_id=utt_id, wav_path=Path(wav_path), text=text)
            )

    if not entries:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")

    return entries


class TimitDataset(TorchDataset):
    """TIMIT dataset that returns ``{"speech", "text"}`` samples.

    The audio is read straight from the TIMIT distribution, which ships NIST
    SPHERE files that ``soundfile`` decodes natively. They are already 16 kHz
    mono, so nothing is resampled and no converted copy is written.

    ``text`` is the IPA reference the builder derived from the ``.PHN`` labels.
    The sample deliberately carries no utterance id, because ESPnet passes the
    whole dictionary onward and unsupported fields break downstream stages; the
    recipe's ``build_output`` uses the item index instead, and the manifest is
    sorted so an index maps back to an id.

    Args:
        recipe_dir: Optional recipe root. When omitted, defaults to the current
            recipe directory inferred from this module.

    Raises:
        FileNotFoundError: If the manifest does not exist.

    Examples:
        >>> dataset = TimitDataset()
        >>> sorted(dataset[0].keys())
        ['speech', 'text']
    """

    def __init__(self, recipe_dir: str | Path | None = None) -> None:
        recipe_root = (
            Path(recipe_dir).resolve()
            if recipe_dir is not None
            else Path(__file__).resolve().parents[1]
        )
        self.data_dir = resolve_data_root(recipe_root)

        builder = TimitBuilder()
        if not builder.is_source_prepared(recipe_dir=recipe_root):
            builder.prepare_source(recipe_dir=recipe_root)
        if not builder.is_built(recipe_dir=recipe_root):
            builder.build(recipe_dir=recipe_root)

        manifest_path = self.data_dir / _DATASET_CFG["manifest_path"]
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        self._entries = _read_manifest(manifest_path)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self._entries[int(idx)]
        array, _sr = sf.read(str(entry.wav_path))
        return {
            "speech": np.asarray(array, dtype=np.float32),
            "text": entry.text,
        }
