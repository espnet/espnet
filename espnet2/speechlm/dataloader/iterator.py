# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Data iterator factory for creating batch iterators in SpeechLM training."""

import json
import logging
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, TypeVar, Union

from torch.utils.data import DataLoader

from espnet2.speechlm.dataloader.batch import batchfy
from espnet2.speechlm.dataloader.dataset import CombinedDataset
from espnet2.speechlm.dataloader.task_conf import TASK_CONFIGS

T = TypeVar("T")


class DataIteratorFactory:
    """Factory for creating data iterators for SpeechLM training.

    This class manages batching, data sharding across GPUs, and provides
    DataLoader instances for training with support for endless epochs.

    Features:
        - Supports multiple tasks and datasets with resampling factors
        - Bucket or pack batching strategies
        - Distributed training with automatic batch synchronization
        - Deterministic shuffling with configurable seeds
        - State saving/loading for training resumption

    Args:
        unregistered_specifier: Space-separated unregistered data specs.
            Format: "task:name:data_json[:factor]"
            Example: "asr:librispeech:train.json:2.0"
        registered_specifier: Space-separated registered data specs.
            Format: "task:name[:factor]"
            Example: "tts:ljspeech:1.5"
        stats_dir: Directory containing statistics files (str or Path).
            Each file should be named "stats_{task}_{data_name}.jsonl"
        collate_fn: Optional collate function for DataLoader.
        loader_state: Optional saved state dict to restore from.
        batchfy_method: Batching method ("bucket" or "pack").
        batch_size: Maximum tokens per batch.
        num_workers: Number of DataLoader workers.
        rank: GPU rank for distributed training (0-indexed).
        world_size: Total number of GPUs in distributed training.
        shuffle: Whether to shuffle batches.
        seed: Random seed for reproducibility.

    Example:
        >>> factory = DataIteratorFactory(
        ...     unregistered_specifier="asr:libri:train.json:2.0",
        ...     registered_specifier="tts:lj:1.0",
        ...     stats_dir="/path/to/stats",
        ...     batch_size=10000,
        ...     shuffle=True,
        ... )
        >>> loader = factory.get_iterator(global_step=0, length=100)
        >>> for batch in loader:
        ...     # Training loop
        ...     pass
    """

    def __init__(
        self,
        unregistered_specifier: str,
        registered_specifier: str,
        stats_dir: Union[str, Path] = None,
        loader_state: Optional[Path] = None,
        collate_fn: Optional[Callable] = None,
        batchfy_method: str = "bucket",
        batch_size: int = 1000,
        num_workers: int = 4,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = False,
        sequential_load: bool = False,
        seed: int = 42,
    ):
        self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.sequential_load = sequential_load
        self.seed = seed

        # Convert stats_dir to Path if it's a string
        if isinstance(stats_dir, str):
            stats_dir = Path(stats_dir)

        # (1) parse data specifier
        cache_unregistered, cache_registered = _parse_data_specifier(
            unregistered_specifier,
            registered_specifier,
        )

        # (2) build dataset
        # Extract (name, data_json) tuples for unregistered datasets
        dataset_unregistered = list(
            set((name, data_json) for _, name, data_json, _ in cache_unregistered)
        )
        # Extract (name,) tuples for registered datasets
        dataset_registered = list(set(name for _, name, _ in cache_registered))
        logging.info(
            f"Building dataset with unregistered={dataset_unregistered}, "
            f"registered={dataset_registered}"
        )
        dataset = CombinedDataset(
            dataset_unregistered,
            dataset_registered,
            num_worker=num_workers,
            rank=rank,
            world_size=world_size,
        )
        logging.info("Dataset construction completed")

        # Store dataset for later use
        self.dataset = dataset

        if self.sequential_load:
            assert self.num_workers == 0, "No multiple workers during collect_stats"

            all_subsets = dataset.get_all_examples()
            self.batched_examples = []
            for entry in cache_registered + cache_unregistered:
                # Extract fields based on tuple structure
                # Registered: (task, name, factor)
                # Unregistered: (task, name, data_json, factor)
                task = entry[0]
                data_name = entry[1]

                required_entries = TASK_CONFIGS[task]["required_entries"]
                dataset.verify_subset_entries(task, data_name, required_entries)

                data_list = all_subsets[data_name]
                data_list = [(task, data_name, example_id) for example_id in data_list]
                # Accumulate all examples as individual batches
                self.batched_examples.extend([[example] for example in data_list])

        elif loader_state is None or (
            loader_state is not None and not loader_state.exists()
        ):
            # (3) build all (task, dataset, example_id) for indexing
            all_subsets = dataset.get_all_examples()
            all_examples = list()
            all_lengths = dict()
            # Process both registered and unregistered datasets
            for entry in cache_registered + cache_unregistered:
                # Extract fields based on tuple structure
                # Registered: (task, name, factor)
                # Unregistered: (task, name, data_json, factor)
                task = entry[0]
                data_name = entry[1]
                factor = entry[-1]  # Last element is always factor

                data_list = all_subsets[data_name]
                data_list = [(task, data_name, example_id) for example_id in data_list]
                # Create deterministic seed for reproducible resampling
                resample_seed = self.seed + hash((task, data_name)) % 100000
                resampled_data_list = _resample(data_list, factor, seed=resample_seed)
                all_examples.extend(resampled_data_list)

                stat_file = stats_dir / f"stats_{task}_{data_name}.jsonl"
                stat_dict = _load_stats(stat_file, task, data_name)
                # Only keep stats for the examples we're using
                filtered_stats = {
                    key: stat_dict[key] for key in data_list if key in stat_dict
                }
                all_lengths.update(filtered_stats)
                logging.info(f"Task={task}, data_name={data_name}, factor={factor}")

            # (4) Build batches
            batched_examples = batchfy(
                all_examples, all_lengths, batch_size, batchfy_method
            )
            self.batched_examples = batched_examples
            logging.info(f"Overall number of batches: {len(batched_examples)}")

            # Only save state if loader_state path was provided
            if loader_state is not None:
                self.save_iterator_state(loader_state)
        else:
            # loader_state is not None and exists
            self.load_iterator_state(loader_state)

    def build_iter(self, global_step: int = 0, length: int = None) -> DataLoader:
        """Get a DataLoader for a specific range of batches.

        Supports endless epochs by wrapping around when batches are
        exhausted. If the requested length exceeds remaining batches,
        it will continue from the beginning.

        Args:
            global_step: Starting batch index (must be non-negative).
            length: Number of batches to include (must be positive).

        Returns:
            DataLoader that iterates over the specified batch range.

        Raises:
            ValueError: If validation fails or no batches available.
        """
        total_batches = len(self.batched_examples)

        if length is None:
            length = total_batches

        # Validate parameters
        if global_step < 0:
            raise ValueError(f"global_step must be non-negative, got {global_step}")
        if length <= 0:
            raise ValueError(f"length must be positive, got {length}")
        if total_batches == 0:
            raise ValueError("No batches available. Cannot create iterator.")

        # Normalize global_step to be within range
        start_idx = global_step % total_batches

        # Build batch subset with wrapping
        batch_subset = [
            self.batched_examples[(start_idx + i) % total_batches]
            for i in range(length)
        ]

        # Shuffle if needed
        if self.shuffle:
            rng = random.Random(self.seed + global_step)
            rng.shuffle(batch_subset)
            logging.info(f"Shuffled batches with seed {self.seed + global_step}")

        # Calculate epoch information for logging
        start_epoch = global_step // total_batches
        end_step = global_step + length
        end_epoch = (end_step - 1) // total_batches

        if start_epoch == end_epoch:
            logging.info(
                f"Created DataLoader with {length} batches "
                f"(epoch {start_epoch}, steps {global_step} to {end_step})"
            )
        else:
            logging.info(
                f"Created DataLoader with {length} batches "
                f"(epochs {start_epoch} to {end_epoch}, "
                f"steps {global_step} to {end_step})"
            )

        # Create DataLoader with batch_sampler
        dataloader = DataLoader(
            self.dataset,
            batch_sampler=batch_subset,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return dataloader

    def save_iterator_state(self, loader_state: str):
        """Save the current state of the iterator to a file.

        Args:
            loader_state: Path to save the iterator state file
        """
        # Save as JSON (nested tuples will be converted to lists)
        state = {"batched_examples": self.batched_examples}

        with open(loader_state, "w") as f:
            json.dump(state, f)

        logging.info(
            f"Saved iterator state to {loader_state} "
            f"with {len(self.batched_examples)} batches"
        )

    def load_iterator_state(self, loader_state: str):
        """Load iterator state from a file.

        Args:
            loader_state: Path to the iterator state file

        Raises:
            FileNotFoundError: If the state file doesn't exist
            KeyError: If required keys are missing in the state file
        """
        with open(loader_state, "r") as f:
            state = json.load(f)

        if "batched_examples" not in state:
            raise KeyError("State file must contain 'batched_examples' key")

        # Convert nested lists back to proper structure:
        # Each batch is a list of tuples (converted from lists by JSON)
        self.batched_examples = [
            [tuple(example) for example in batch] for batch in state["batched_examples"]
        ]

        logging.info(
            f"Loaded iterator state from {loader_state} "
            f"with {len(self.batched_examples)} batches"
        )


def _parse_data_specifier(
    task_data_factors: str,
    task_registered_data_factors: str,
) -> Tuple[List[Tuple], List[Tuple]]:
    """Parse data specifier strings into structured tuples.

    Args:
        task_data_factors: Space-separated unregistered data specifiers.
            Format: "task:name:data_json[:factor]"
            Example: "asr:librispeech:train.json:2.0"
        task_registered_data_factors: Space-separated registered data
            specifiers. Format: "task:name[:factor]"
            Example: "tts:ljspeech:1.5"

    Returns:
        Tuple of (unregistered_list, registered_list) where:
        - unregistered_list: List of (task, name, data_json, factor)
        - registered_list: List of (task, name, factor)

    Raises:
        ValueError: If specifier format is invalid.
    """
    cache_unregistered = []
    if task_data_factors.strip():
        for entry in task_data_factors.split():
            parts = entry.split(":")
            if len(parts) == 4:
                task, name, data_json, factor = parts
                factor = float(factor)
            elif len(parts) == 3:
                task, name, data_json = parts
                factor = 1.0
            else:
                raise ValueError(
                    f"Invalid unregistered specifier '{entry}'. "
                    f"Expected format: 'task:name:data_json[:factor]'"
                )
            cache_unregistered.append((task, name, data_json, factor))

    cache_registered = []
    if task_registered_data_factors.strip():
        for entry in task_registered_data_factors.split():
            parts = entry.split(":")
            if len(parts) == 3:
                task, name, factor = parts
                factor = float(factor)
            elif len(parts) == 2:
                task, name = parts
                factor = 1.0
            else:
                raise ValueError(
                    f"Invalid registered specifier '{entry}'. "
                    f"Expected format: 'task:name[:factor]'"
                )
            cache_registered.append((task, name, factor))

    return cache_unregistered, cache_registered


def _load_stats(
    stat_file: Path, task: str, data_name: str
) -> Dict[Tuple[str, str, str], int]:
    """Load statistics from a JSONL file.

    Each line in the file should be a JSON object mapping example IDs
    to their lengths (in tokens or frames).

    Args:
        stat_file: Path to the JSONL statistics file.
        task: Task name to use in the result keys.
        data_name: Dataset name to use in the result keys.

    Returns:
        Dictionary mapping (task, data_name, example_id) to length.

    Raises:
        FileNotFoundError: If stat_file does not exist.
        json.JSONDecodeError: If file contains invalid JSON.
        ValueError: If length values are not valid integers.
    """
    if not stat_file.exists():
        raise FileNotFoundError(f"Statistics file not found: {stat_file}")

    result = {}
    try:
        with open(stat_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    raise json.JSONDecodeError(
                        f"Invalid JSON at line {line_num} in {stat_file}", e.doc, e.pos
                    )

                if not isinstance(data, dict):
                    raise ValueError(f"Line {line_num} in {stat_file} is not a dict")

                for example_id, length in data.items():
                    if not isinstance(length, (int, float)):
                        raise ValueError(
                            f"Invalid length value for '{example_id}' "
                            f"at line {line_num}: {length}"
                        )
                    result[(task, data_name, example_id)] = int(length)

    except Exception as e:
        logging.error(f"Failed to load stats from {stat_file}: {e}")
        raise

    logging.info(f"Loaded {len(result)} examples from {stat_file}")
    return result


def _resample(lst: List[T], factor: float, seed: int = 42) -> List[T]:
    """Resample a list by a given factor.

    For integer factors, simply repeats the list. For non-integer
    factors, repeats the whole list for the integer part and randomly
    samples the fractional part using a deterministic seed.

    Args:
        lst: List to resample.
        factor: Resampling factor. Must be positive.
            - factor=1.0: returns original list
            - factor=2.0: returns list repeated twice
            - factor=1.5: returns list once + 50% random sample
        seed: Random seed for reproducible sampling. Default: 42.

    Returns:
        Resampled list. Returns empty list if input is empty.

    Raises:
        ValueError: If factor <= 0.

    Examples:
        >>> _resample([1, 2, 3], 2.0)
        [1, 2, 3, 1, 2, 3]
        >>> _resample([1, 2, 3], 1.5, seed=42)
        [1, 2, 3, 2, 1]  # Deterministic with same seed
    """
    if factor <= 0:
        raise ValueError(f"Resampling factor must be positive, got {factor}")

    if not lst:
        return []  # Empty list resampled is still empty

    # Integer factor: simple repetition
    if factor.is_integer():
        return lst * int(factor)

    # Non-integer factor: repeat whole + sample fractional part
    result = []

    # Add full copies for the integer part
    int_part = int(factor)
    for _ in range(int_part):
        result.extend(lst)

    # Add random sample for the fractional part (deterministic)
    frac_part = factor - int_part
    num_samples = int(frac_part * len(lst))
    if num_samples > 0:
        # Use local Random instance for reproducibility
        rng = random.Random(seed)
        residual = rng.sample(lst, num_samples)
        result.extend(residual)

    return result
