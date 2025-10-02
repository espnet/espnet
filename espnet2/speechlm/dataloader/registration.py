#!/usr/bin/env python3
"""Dataset registration mechanism for ESPnet SpeechLM."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple


# Global registries
_datasets: Dict[str, str] = {}

# Hardcoded registration folders
DATASET_REGISTRATION_FOLDERS: List[str] = [
    # Add your dataset registration folders here
    # Example: "/path/to/dataset/configs"
]


# Dataset registration
def register_dataset(name: str, json_file: str) -> None:
    """Register a dataset JSON file."""
    if name in _datasets:
        logging.warning(f"Dataset '{name}' already registered with path: {_datasets[name]}, skipping new path: {json_file}")
        return
    _datasets[name] = json_file


def get_dataset(name: str) -> str:
    """Get dataset JSON file path."""
    if name not in _datasets:
        raise KeyError(f"Dataset '{name}' not registered")
    return _datasets[name]


def list_datasets():
    """List all registered dataset names."""
    return list(_datasets.keys())


# Auto-registration from folders
def auto_register_datasets_from_folders() -> None:
    """Automatically register datasets from hardcoded folders.

    Scans all JSON files in DATASET_REGISTRATION_FOLDERS and registers datasets.
    Expected JSON format: {"data_name1": "path1.json", "data_name2": "path2.json"}
    """
    for folder in DATASET_REGISTRATION_FOLDERS:
        folder_path = Path(folder)
        if not folder_path.exists():
            continue

        # Recursively find all JSON files
        for json_file in folder_path.rglob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    config = json.load(f)

                if config and isinstance(config, dict):
                    for data_name, data_path in config.items():
                        if isinstance(data_path, str):
                            register_dataset(data_name, data_path)
                    logging.info(f"Successfully loaded dataset registration from {json_file}")
            except Exception as e:
                logging.warning(f"Failed to load dataset registration from {json_file}: {e}")

auto_register_datasets_from_folders()
