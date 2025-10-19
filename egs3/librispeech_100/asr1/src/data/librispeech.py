"""Hugging Face dataset wrapper for the LibriSpeech 100h recipe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
from datasets import Audio, Dataset, DatasetDict, load_from_disk
from torch.utils.data import Dataset as TorchDataset


@dataclass
class LibriSpeechDatasetConfig:
    """Configuration options for :class:`LibriSpeechDataset`."""

    data_dir: str | Path = "data"
    split: str | None = None
    sample_rate: int = 16_000
    return_text: bool = True
    return_audio: bool = True
    return_utt_id: bool = True


class LibriSpeechDataset(TorchDataset):
    """Minimal dataset for LibriSpeech prepared with ``create_dataset.py``."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        split: str | None = None,
        sample_rate: int = 16_000,
        return_text: bool = True,
        return_audio: bool = True,
        return_utt_id: bool = True,
    ) -> None:
        cfg = LibriSpeechDatasetConfig(
            data_dir=data_dir,
            split=split,
            sample_rate=sample_rate,
            return_text=return_text,
            return_audio=return_audio,
            return_utt_id=return_utt_id,
        )

        data_root = Path(cfg.data_dir)
        if not data_root.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {data_root.resolve().as_posix()}"
            )
        if cfg.split is None:
            raise ValueError("Please specify the Hugging Face dataset split name.")

        dataset = load_from_disk(str(data_root))
        if isinstance(dataset, DatasetDict):
            if cfg.split not in dataset:
                raise ValueError(
                    f"Split '{cfg.split}' not found. Available: {list(dataset.keys())}"
                )
            dataset = dataset[cfg.split]
        elif not isinstance(dataset, Dataset):
            raise TypeError(
                "Expected a Dataset or DatasetDict saved with create_dataset.py"
            )

        if "audio" in dataset.column_names:
            dataset = dataset.cast_column("audio", Audio(sampling_rate=cfg.sample_rate))

        self.dataset: Dataset = dataset
        self.sample_rate = int(cfg.sample_rate)
        self.return_text = bool(cfg.return_text)
        self.return_audio = bool(cfg.return_audio)
        self.return_utt_id = bool(cfg.return_utt_id)

    def __len__(self) -> int:
        return len(self.dataset)

    def _build_example(self, idx: int, item: Dict) -> Dict:
        example: Dict[str, object] = {}
        if self.return_utt_id:
            utt_id = item.get("id") or item.get("utt_id")
            if not utt_id:
                audio_path = item.get("audio", {}).get("path")
                utt_id = Path(audio_path).stem if audio_path else f"utt{idx}"
            example["utt_id"] = str(utt_id)

        if self.return_audio and "audio" in item:
            array = item["audio"].get("array")
            if array is None:
                raise RuntimeError(
                    "Audio column is not decoded. Please run create_dataset.py first."
                )
            example["speech"] = np.asarray(array, dtype=np.float32)
            example["sample_rate"] = int(item["audio"].get("sampling_rate", self.sample_rate))

        if self.return_text and "text" in item:
            example["text"] = item["text"]
        return example

    def __getitem__(self, idx: int) -> Dict:
        item = self.dataset[idx]
        return self._build_example(idx, item)

    def get_text(self, idx: int) -> str:
        item = self.dataset[idx]
        text = item.get("text")
        if text is None:
            raise KeyError("Text column is missing in the dataset.")
        return text