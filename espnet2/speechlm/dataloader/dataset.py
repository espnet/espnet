#!/usr/bin/env python3
import json
from typing import Any, Dict, List, Tuple

from torch.utils.data import Dataset

from espnet2.speechlm.dataloader.multimodal_loader import (
    LhotseAudioReader,
    TextReader,
)
from espnet2.speechlm.dataloader.registration import get_dataset


class ESPnetSpeechLMDataset(Dataset):
    """ESPnet Speech Language Model Dataset.

    Args:
        json_file: Path to dataset JSON created by prepare_dataset_json.py
        rank: Process rank for distributed training (default: 0)
        world_size: Total number of processes (default: 1)
    """

    def __init__(self, json_file: str, rank: int = 0, world_size: int = 1):
        # Load JSON
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Get data entries and samples
        data_entries = data["data_entry"]
        all_samples = data["samples"]

        # Filter samples for this rank
        self.samples = all_samples[rank::world_size]

        # Build readers
        self.readers: Dict[str, Any] = {}

        for entry in data_entries:
            name = entry["name"]
            path = entry["path"]
            reader_type = entry["reader"]

            # Create appropriate reader with valid_ids for this rank
            if reader_type == "lhotse_audio":
                self.readers[name] = LhotseAudioReader(path, valid_ids=self.samples)
            elif reader_type == "text":
                self.readers[name] = TextReader(path, valid_ids=self.samples)
            else:
                raise ValueError(f"Unknown reader type: {reader_type}")

    @property
    def entries(self) -> List[str]:
        """Return list of all data entry names."""
        return list(self.readers.keys())

    @property
    def sample_ids(self) -> List[str]:
        """Return list of all sample IDs."""
        return self.samples

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, sample_id: str) -> Dict[str, Any]:
        """Get item by sample_id.

        Args:
            sample_id: The sample ID to retrieve

        Returns:
            Dictionary with keys from data entry names
        """
        # Collect data from all readers
        item = {}
        for name, reader in self.readers.items():
            item[name] = reader[sample_id]

        return item


class CombinedESPnetSpeechLMDataset(Dataset):
    """Combined ESPnet Speech Language Model Dataset.

    Combines multiple registered datasets into a single dataset.

    Args:
        dataset_names: List of registered dataset names
        rank: Process rank for distributed training (default: 0)
        world_size: Total number of processes (default: 1)
    """

    def __init__(
        self,
        dataset_names: List[str],
        rank: int = 0,
        world_size: int = 1,
    ):
        self.datasets: Dict[str, ESPnetSpeechLMDataset] = {}

        # Load all datasets from registry
        for name in dataset_names:
            json_path = get_dataset(name)
            self.datasets[name] = ESPnetSpeechLMDataset(
                json_path, rank=rank, world_size=world_size
            )

    @property
    def dataset_names(self) -> List[str]:
        """Return list of all dataset names."""
        return list(self.datasets.keys())

    def get_all_examples(self) -> List[Tuple[str, List[str]]]:
        """Return list of all examples as (dataset_name, example_id_list) tuples.

        Returns:
            List of (dataset_name, example_id_list) tuples
        """
        examples = []
        for dataset_name, dataset in self.datasets.items():
            examples.append((dataset_name, dataset.sample_ids))
        return examples

    def __len__(self) -> int:
        """Return total number of samples across all datasets."""
        return sum(len(dataset) for dataset in self.datasets.values())

    def __getitem__(self, key: Tuple[str, str]) -> Dict[str, Any]:
        """Get item by (dataset_name, sample_id).

        Args:
            key: Tuple of (dataset_name, sample_id)

        Returns:
            Dictionary with keys from data entry names
        """
        dataset_name, sample_id = key
        return self.datasets[dataset_name][sample_id]
