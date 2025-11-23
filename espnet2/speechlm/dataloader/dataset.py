#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Dataset implementation for multimodal data loading in SpeechLM training."""

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import yaml
from torch.utils.data import Dataset

from espnet2.speechlm.dataloader.multimodal_loader.audio_loader import LhotseAudioReader
from espnet2.speechlm.dataloader.multimodal_loader.text_loader import TextReader

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


def _load_dataset_worker(args):
    """Worker function for multiprocessing dataset loading.

    Args:
        args: Tuple of (dataset_name, json_path, rank, world_size)

    Returns:
        Tuple of (dataset_name, SingleDataset instance, dataset_length)
    """
    dataset_name, json_path, rank, world_size = args
    dataset = SingleDataset(json_path, rank=rank, world_size=world_size)
    return dataset_name, dataset, len(dataset)


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
            reader_class = reader_types[reader_type]
            self.readers[name] = reader_class(path, valid_ids=self.samples)

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
        num_worker: Number of parallel workers for loading datasets.
        rank: Process rank for distributed training (default: 0)
        world_size: Total number of processes (default: 1)
    """

    def __init__(
        self,
        datasets: List[Tuple[str, str]] = [],
        registered_datasets: List[str] = [],
        num_worker: int = 1,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.datasets: Dict[str, SingleDataset] = {}

        # Step 1: Collect all (dataset_name, json_path) pairs
        dataset_paths = []
        seen_names = set()

        # Collect from direct paths
        for dataset_name, json_path in datasets:
            if dataset_name in seen_names:
                raise ValueError(f"Duplicate dataset name: {dataset_name}")
            dataset_paths.append((dataset_name, json_path))
            seen_names.add(dataset_name)

        # Collect from registry
        registry_data = self._load_registry()
        for dataset_name in registered_datasets:
            if dataset_name in seen_names:
                raise ValueError(f"Duplicate dataset name: {dataset_name}")
            if dataset_name not in registry_data:
                raise ValueError(
                    f"Dataset '{dataset_name}' not found in registry. "
                    f"Available datasets: {list(registry_data.keys())}"
                )
            dataset_paths.append((dataset_name, registry_data[dataset_name]))
            seen_names.add(dataset_name)

        # Step 2: Load all datasets in parallel using multiprocessing
        # Prepare arguments for multiprocessing
        worker_args = [(name, path, rank, world_size) for name, path in dataset_paths]

        # Use ProcessPoolExecutor for parallel loading
        max_workers = min(num_worker, len(dataset_paths))
        max_workers = max(1, max_workers)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all loading tasks
            futures = [
                executor.submit(_load_dataset_worker, args) for args in worker_args
            ]

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    dataset_name, dataset, dataset_len = future.result()
                    self.datasets[dataset_name] = dataset
                    logging.info(
                        f"Loaded dataset [{dataset_name}]. "
                        f"Local dataset size: [{dataset_len}]."
                    )
                except Exception as e:
                    raise RuntimeError(f"Failed to load dataset: {e}") from e

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

    def verify_subset_entries(self, task, data_name, required_entries):
        """Verify that a dataset contains all required entries for a task."""
        # Get the dataset's available entries
        entries = self.datasets[data_name].entries
        # Check each required entry exists in the dataset
        for e in required_entries:
            if e not in entries:
                raise ValueError(
                    f"Task {task} requires entry {e} "
                    f"but is missing in dataset: {data_name}"
                )

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
        _, dataset_name, sample_id = key
        return key, self.datasets[dataset_name][sample_id]
