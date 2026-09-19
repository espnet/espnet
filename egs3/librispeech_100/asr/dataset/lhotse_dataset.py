"""LibriSpeech 100h dataset implementation with lhotse."""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from lhotse import MonoCut
from torch.utils.data import Dataset as TorchDataset

from egs3.librispeech_100.asr.dataset.lhotse_builder import (
    LibriSpeech100LhotseBuilder,
    resolve_source_root,
)
from espnet3.utils.config_utils import load_config_with_defaults

_CONFIG_RESOURCE = resources.files(__package__).joinpath("lhotse_config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
_DATASET_CFG = _CONFIG["dataset"]

_KNOWN_SPLITS = {str(split) for split in _DATASET_CFG["supported_splits"]}


class LibriSpeech100LhotseDataset(TorchDataset):
    """Torch dataset that reads LibriSpeech from the cuts built from lhotse manifests.
        Here the assumption is that each cut has one supervision

    Args:
        split: LibriSpeech split directory name such as ``train-clean-100``.
        recipe_dir: Optional recipe root. When omitted, defaults to the current
            recipe directory inferred from this module.
        source_dir: Optional LibriSpeech parent/root override.

    Raises:
        ValueError: If ``split`` is unknown.
        FileNotFoundError: If the resolved source root or split directory does
            not exist.
        RuntimeError: If no transcript/audio pairs are found for the split.

    Examples:
        >>> dataset = LibriSpeech100LhotseDataset(split="train-clean-100")
        >>> sample = dataset[0]
        >>> sorted(sample.keys())
        ['speech', 'text', 'utt_id']
    """

    def __init__(
        self,
        split: str,
        recipe_dir: str | Path | None = None,
        source_dir: str | Path | None = None,
        dataset_dir: str | Path | None = None,
        is_stage_collect_stats: bool = False,
    ) -> None:
        self.split = str(split)
        self.is_stage_collect_stats = is_stage_collect_stats

        if self.split not in _KNOWN_SPLITS:
            known = ", ".join(sorted(_KNOWN_SPLITS))
            raise ValueError(f"Unknown split '{self.split}'. Expected one of: {known}")

        recipe_dir = (
            Path(recipe_dir).resolve()
            if recipe_dir is not None
            else Path(__file__).resolve().parents[1]
        )

        recipe_root = Path(recipe_dir).resolve()
        self.librispeech_root = resolve_source_root(
            recipe_root,
            source_dir=source_dir,
        )
        split_dir = self.librispeech_root / self.split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        lhotse_builder = LibriSpeech100LhotseBuilder()
        lhotse_builder.build(recipe_dir=recipe_dir, dataset_dir=dataset_dir)

        self._cuts = lhotse_builder.load_cutsets(split=split, dataset_dir=dataset_dir)
        self._cut_id_dict = {cut.id: cut for cut in self._cuts}

    def _get_cut(self, idx: [int, str]) -> MonoCut:
        if isinstance(idx, int):
            return self._cuts[idx]
        elif idx.isdigit():
            return self._cuts[int(idx)]
        elif idx in self._cut_id_dict:
            return self._cut_id_dict[idx]
        else:
            raise ValueError("getitem accepts either an int index or an utterance id")

    def __len__(self) -> int:
        return len(self._cuts)

    def __getitem__(self, idx: [int, str]) -> dict[str, Any]:

        cut = self._get_cut(idx)

        sample = {
            "speech": np.asarray(cut.load_audio(), dtype=np.float32).squeeze(),
            "text": cut.supervisions[0].text,
        }

        if self.is_stage_collect_stats:
            sample["utt_id"] = cut.id

        return sample
