#!/usr/bin/env python3
import json
import logging
import os
from typing import Any, Dict, List, Tuple

import yaml
from torch.utils.data import Dataset

from espnet2.speechlm.dataloader.multimodal_loader.text_loader import TextReader
from espnet2.speechlm.dataloader.multimodal_loader.audio_loader import LhotseAudioReader

logger = logging.getLogger(__name__)

reader_types = {
    "lhotse_audio": LhotseAudioReader,
    "text": TextReader,
}

# TODO(Jinchuan): revisit the CPU memory usage for large-scale training. Check official
# information as follow:
# After several iterations, the loader worker processes will consume the same amount of
# CPU memory as the parent process for all Python objects in the parent process which
# are accessed from the worker processes. This can be problematic if the Dataset
# contains a lot of data (e.g., you are loading a very large list of filenames at
# Dataset construction time) and/or you are using a lot of workers (overall memory
# usage is number of workers * size of parent process). The simplest workaround is
# to replace Python objects with non-refcounted representations such as Pandas, Numpy
#  or PyArrow objects. Check out issue #13246 for more details on why this occurs and
#  example code for how to workaround these problems.


class SingleDataset(Dataset):
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
            if reader_type not in reader_types:
                raise ValueError(f"Unknown reader type: {reader_type}")
            reader_type = reader_types[reader_type]
            self.readers[name] = reader_type(path, valid_ids=self.samples)

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


class CombinedDataset(Dataset):
    """Combined ESPnet Speech Language Model Dataset.

    Combines multiple datasets from both direct paths and registered datasets.

    Args:
        datasets: List of (name, json_path) tuples for direct dataset paths
            (default: [])
        registered_datasets: List of registered dataset names to look up in
            registry (default: [])
        rank: Process rank for distributed training (default: 0)
        world_size: Total number of processes (default: 1)
    """

    def __init__(
        self,
        datasets: List[Tuple[str, str]] = [],
        registered_datasets: List[str] = [],
        rank: int = 0,
        world_size: int = 1,
    ):
        self.datasets: Dict[str, SingleDataset] = {}

        # Load datasets from direct paths
        for dataset_name, json_path in datasets:
            if dataset_name in self.datasets:
                raise ValueError(f"Duplicate dataset name: {dataset_name}")
            single_dataset = SingleDataset(json_path, rank=rank, world_size=world_size)
            self.datasets[dataset_name] = single_dataset
            logging.info(
                f"Loaded dataset [{dataset_name}] at {json_path}. "
                f"Local dataset size: [{len(single_dataset)}]."
            )

        # Load datasets from registry
        registry_data = self._load_registry()

        # TODO(Jinchuan): use multi-processing to initialize single datasets,
        # which would accelerate the initializaiton.
        for dataset_name in registered_datasets:
            if dataset_name in registry_data:
                if dataset_name in self.datasets:
                    raise ValueError(f"Duplicate dataset name: {dataset_name}")
                json_path = registry_data[dataset_name]
                single_dataset = SingleDataset(
                    json_path, rank=rank, world_size=world_size
                )
                self.datasets[dataset_name] = single_dataset
                logging.info(
                    f"Loaded dataset [{dataset_name}] at {json_path}. "
                    f"Local dataset size: [{len(single_dataset)}]."
                )
            else:
                raise ValueError(
                    f"Dataset '{dataset_name}' not found in registry. "
                    f"Available datasets: {list(registry_data.keys())}"
                )

    def _load_registry(self) -> Dict[str, str]:
        """Load and merge registry files from ESPNET_DATASET_REGISTRY env variable.

        Returns:
            Dictionary mapping dataset names to JSON file paths
        """
        registry_data = {}

        # Get registry paths from environment variable
        registry_env = os.environ.get("ESPNET_DATASET_REGISTRY", "")
        if not registry_env:
            return registry_data

        # Split by : and filter out empty strings
        registry_paths = [
            path.strip() for path in registry_env.split(":") if path.strip()
        ]

        for registry_path in registry_paths:
            if not os.path.exists(registry_path):
                logger.warning(f"Registry file not found: {registry_path}")
                continue

            try:
                with open(registry_path, "r") as f:
                    registry_content = yaml.safe_load(f)

                    # Extract dataset names and paths from the registry
                    for dataset_name, dataset_info in registry_content.items():
                        if isinstance(dataset_info, dict) and "path" in dataset_info:
                            # Check for duplicate dataset names across registries
                            if dataset_name in registry_data:
                                logger.warning(
                                    f"Dataset '{dataset_name}' already exists, "
                                    f"overriding with entry from {registry_path}"
                                )
                            registry_data[dataset_name] = dataset_info["path"]
            except Exception as e:
                logger.error(f"Error loading registry file {registry_path}: {e}")
                continue

        return registry_data

    @property
    def dataset_names(self) -> List[str]:
        """Return list of all dataset names."""
        return list(self.datasets.keys())

    def get_all_examples(self) -> Dict[str, List[str]]:
        """Return all examples as a dictionary mapping dataset names to sample IDs.

        Returns:
            Dictionary mapping dataset names to lists of sample IDs
        """
        examples = {}
        for dataset_name, dataset in self.datasets.items():
            examples[dataset_name] = dataset.sample_ids
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
