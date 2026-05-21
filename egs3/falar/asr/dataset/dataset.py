"""FalAR dataset implementation backed by Hugging Face datasets."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Audio, concatenate_datasets, load_dataset
from torch.utils.data import Dataset as TorchDataset

from espnet3.utils.config_utils import load_config_with_defaults

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
_BUILDER_CFG = _CONFIG["builder"]
_DATASET_CFG = _CONFIG["dataset"]

_HF_DATASET = str(_BUILDER_CFG["hf_dataset"])
_AUDIO_COLUMN = str(_DATASET_CFG["audio_column"])
_TEXT_COLUMN = str(_DATASET_CFG["text_column"])
_ID_COLUMN = str(_DATASET_CFG["id_column"])
_LANGUAGE_PREFIX = str(_DATASET_CFG["language_prefix"])
_TASK_PREFIX = str(_DATASET_CFG["task_prefix"])
_NOTIMESTAMPS_SYMBOL = str(_DATASET_CFG["notimestamps_symbol"])

_SPLIT_ALIASES = {
    str(alias): str(target)
    for alias, target in _DATASET_CFG.get("split_aliases", {}).items()
}
_TRAIN_SHARDS = [str(split) for split in _DATASET_CFG["train_shards"]]
_KNOWN_SPLITS = {
    "train",
    *_TRAIN_SHARDS,
    *_SPLIT_ALIASES,
    str(_DATASET_CFG["dev_split"]),
    str(_DATASET_CFG["test_split"]),
}


def _resolve_split(split: str | None) -> list[str]:
    """Resolve recipe split aliases to Hugging Face split names."""
    split_name = "train" if split is None else str(split)
    split_name = _SPLIT_ALIASES.get(split_name, split_name)
    if split_name == "train":
        return _TRAIN_SHARDS
    if split_name not in _KNOWN_SPLITS:
        known = ", ".join(sorted(_KNOWN_SPLITS))
        raise ValueError(
            f"Unknown FalAR split '{split_name}'. Expected one of: {known}"
        )
    return [split_name]


class FalarDataset(TorchDataset):
    """FalAR ASR dataset for ESPnet3 recipes.

    This dataset reads ``inesc-id/FalAR`` with Hugging Face ``datasets`` and
    returns ESPnet-style sample dictionaries. Use recipe-level split names such
    as ``train``, ``valid``, or ``test`` in YAML. The dataset maps them to the
    actual Hugging Face split layout.

    Args:
        split: Recipe split name. ``train`` loads all ``train_0`` to
            ``train_15`` shards. ``valid`` and ``validation`` load ``dev``.
            ``test`` loads ``test``.
        cache_dir: Optional Hugging Face cache directory.
        ratio: Fraction of the resolved split to keep from the front. This is
            useful for smoke tests and small experiments.

    Returns:
        Sample dictionaries with ``utt_id``, ``speech``, ``text``, ``text_ctc``,
        and ``text_prev`` keys.

    Raises:
        ValueError: If ``ratio`` is outside ``(0, 1]`` or ``split`` is unknown.

    Notes:
        ``text`` includes the OWSM-style Portuguese ASR prefix. ``text_ctc``
        keeps the plain normalized transcript.

    Examples:
        ```python
        dataset = FalarDataset(split="valid", ratio=0.01)
        sample = dataset[0]
        assert sample["utt_id"]
        ```

        YAML dataset references can use the recipe-local module:
        ```yaml
        dataset:
          train:
            - data_src_args:
                split: train
          valid:
            - data_src_args:
                split: valid
        ```
    """

    def __init__(
        self,
        split: str | None = "train",
        cache_dir: str | Path | None = None,
        ratio: float = 1.0,
    ) -> None:
        if not (0 < ratio <= 1.0):
            raise ValueError("ratio must be in the range (0, 1].")

        hf_splits = _resolve_split(split)
        datasets = [
            load_dataset(
                _HF_DATASET,
                split=hf_split,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
                download_mode="reuse_dataset_if_exists",
            )
            for hf_split in hf_splits
        ]
        self.dataset = (
            datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)
        )
        self.split = "train" if split is None else str(split)
        self.hf_splits = hf_splits

        if ratio < 1.0:
            keep = max(1, int(len(self.dataset) * ratio))
            self.dataset = self.dataset.select(range(keep))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[int(idx)]
        transcript = str(item[_TEXT_COLUMN]).strip()
        utt_id = str(item[_ID_COLUMN]).strip()
        text = (
            f"{_LANGUAGE_PREFIX}{_TASK_PREFIX}"
            f"{_NOTIMESTAMPS_SYMBOL} {transcript}"
        )
        speech = item[_AUDIO_COLUMN].get_all_samples().data

        return {
            "speech": speech[0].numpy().astype(np.float32),
            "text": text,
            "text_ctc": transcript,
            "text_prev": "<na>",
        }
