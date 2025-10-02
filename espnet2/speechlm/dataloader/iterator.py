#!/usr/bin/env python3
"""Iterator building for ESPnet SpeechLM datasets."""

import random
from typing import List

from espnet2.speechlm.dataloader.dataset import CombinedESPnetSpeechLMDataset
from espnet2.speechlm.dataloader.registration import register_dataset


def parse_dataset_spec(spec: str, has_path: bool = False):
    """Parse dataset specification string.

    Args:
        spec: Dataset specification string
        has_path: If True, expects "data_name:path:task:factor" or "data_name:path:task"
                  If False, expects "data_name:task:factor" or "data_name:task"

    Returns:
        Tuple of parsed values based on has_path flag
    """
    parts = spec.split(":")

    if has_path:
        # Format: data_name:path:task:factor or data_name:path:task
        if len(parts) == 4:
            data_name, path, task, factor = parts
            return data_name, path, task, float(factor)
        elif len(parts) == 3:
            data_name, path, task = parts
            return data_name, path, task, 1.0
        else:
            raise ValueError(f"Invalid unregistered dataset spec: {spec}")
    else:
        # Format: data_name:task:factor or data_name:task
        if len(parts) == 3:
            data_name, task, factor = parts
            return data_name, task, float(factor)
        elif len(parts) == 2:
            data_name, task = parts
            return data_name, task, 1.0
        else:
            raise ValueError(f"Invalid registered dataset spec: {spec}")


def build_iterator(
    registered_datasets: List[str] = None,
    unregistered_datasets: List[str] = None,
    rank: int = 0,
    world_size: int = 1,
):
    """Build iterator from registered and unregistered datasets.

    Args:
        registered_datasets: List of "data_name:task:factor" or "data_name:task" strings
        unregistered_datasets: List of "data_name:path:task:factor" or "data_name:path:task" strings
        rank: Process rank for distributed training (default: 0)
        world_size: Total number of processes (default: 1)

    Returns:
        Tuple of (CombinedESPnetSpeechLMDataset, all_examples)
        where all_examples is List[(data_name, task, example_id)]
    """
    if registered_datasets is None:
        registered_datasets = []
    if unregistered_datasets is None:
        unregistered_datasets = []

    dataset_task_factors = {}  # (data_name, task) -> factor

    # Parse unregistered datasets and register them
    for spec in unregistered_datasets:
        data_name, path, task, factor = parse_dataset_spec(spec, has_path=True)
        register_dataset(data_name, path)  # Handles duplicates gracefully
        dataset_task_factors[(data_name, task)] = factor

    # Parse registered datasets
    for spec in registered_datasets:
        data_name, task, factor = parse_dataset_spec(spec, has_path=False)
        dataset_task_factors[(data_name, task)] = factor

    # Get unique dataset names from keys
    all_dataset_names = list(set(dn for dn, _ in dataset_task_factors.keys()))

    # Build combined dataset
    combined_dataset = CombinedESPnetSpeechLMDataset(
        dataset_names=all_dataset_names,
        rank=rank,
        world_size=world_size,
    )

    # Get all examples and apply factors
    all_examples = []

    for data_name, example_id_list in combined_dataset.get_all_examples():
        # Find all tasks for this dataset
        for (dn, task), factor in dataset_task_factors.items():
            if dn != data_name:
                continue

            num_repeats = int(factor)
            remainder = factor - num_repeats

            for example_id in example_id_list:
                # Add full repeats
                for _ in range(num_repeats):
                    all_examples.append((data_name, task, example_id))
                # Add partial repeat probabilistically
                if remainder > 0 and random.random() < remainder:
                    all_examples.append((data_name, task, example_id))

    return combined_dataset, all_examples
