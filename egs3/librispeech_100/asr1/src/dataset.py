"""Hugging Face dataset wrapper for the LibriSpeech 100h recipe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import soundfile as sf
from datasets import Audio, Dataset, DatasetDict, load_from_disk
from torch.utils.data import Dataset as TorchDataset


@dataclass
class LibriSpeechDatasetConfig:
    """Configuration options for :class:`LibriSpeechDataset`."""

    data_dir: str | Path = "data"
    split: str | None = None
    sample_rate: int = 16000
    return_text: bool = True
    return_speech: bool = True


class LibriSpeechDataset(TorchDataset):
    """Minimal dataset for LibriSpeech prepared with ``create_dataset.py``."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        split: str | None = None,
        sample_rate: int = 16000,
        return_text: bool = True,
        return_speech: bool = True,
    ) -> None:
        cfg = LibriSpeechDatasetConfig(
            data_dir=data_dir,
            split=split,
            sample_rate=sample_rate,
            return_text=return_text,
            return_speech=return_speech,
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

        self.dataset: Dataset = dataset
        self.sample_rate = int(cfg.sample_rate)
        self.return_text = bool(cfg.return_text)
        self.return_speech = bool(cfg.return_speech)

    def __len__(self) -> int:
        return len(self.dataset)

    def _build_example(self, idx: int, item: Dict) -> Dict:
        example: Dict[str, object] = {}
        speech_path = item.get("speech", {}).get("path")

        if self.return_speech:
            array = sf.read(speech_path)[0]
            example["speech"] = np.asarray(array, dtype=np.float32)

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
