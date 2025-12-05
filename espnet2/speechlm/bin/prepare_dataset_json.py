#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Script for preparing dataset JSON from multimodal data sources."""

import argparse
import json
import logging
from pathlib import Path

from espnet2.speechlm.dataloader.task_conf import SUPPORTED_ENTRIES
from espnet2.speechlm.dataloader.multimodal_loader import ALL_DATA_LOADERS


def validate_triplet(triplet: str):
    """Validate and parse a name,path,reader triplet.

    Args:
        triplet: String in format "name,path,reader"

    Returns:
        Tuple of (name, path, reader) where path is absolute

    Raises:
        ValueError: If triplet is invalid
    """
    parts = triplet.split(",")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid triplet format: {triplet} (expected 3 comma-separated parts)"
        )

    name, path, reader = parts

    # Validate name (audio1, audio2, ... or text1, text2, ... or dialogue)
    if name not in SUPPORTED_ENTRIES:
        raise ValueError(f"Invalid entry name {name}")

    # Convert to Path and check existence
    path_obj = Path(path)
    if not path_obj.exists():
        raise ValueError(f"Path does not exist: {path}")

    # Convert to absolute path
    absolute_path = str(path_obj.resolve())

    # Validate reader
    if reader not in ALL_DATA_LOADERS:
        raise ValueError(
            f"Invalid reader '{reader}': must be {', '.join(ALL_DATA_LOADERS.keys())}"
        )

    return name, absolute_path, reader


def prepare_dataset_json(
    triplets: list,
    output_json: str,
    log_level: str,
):
    """Prepare dataset JSON from multiple data sources.

    Args:
        triplets: List of name,path,reader triplets
        output_json: Output JSON file path
        log_level: Logging level
    """
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    # Parse and validate triplets
    triplet_info = []
    data_sources = {}

    for triplet in triplets:
        name, path, reader = validate_triplet(triplet)

        logging.info(f"Loading {name} from {path} using {reader} reader")

        # Store triplet information
        triplet_info.append({"name": name, "path": path, "reader": reader})

        # Create appropriate reader
        reader_class = ALL_DATA_LOADERS[reader]
        data_sources[name] = reader_class(path)

    # Find valid samples (those that exist in ALL data sources)
    if not data_sources:
        raise ValueError("No data sources provided")

    # Start with IDs from first source
    valid_ids = set(list(data_sources.values())[0].keys())

    # Intersect with IDs from all other sources
    for reader in data_sources.values():
        valid_ids &= set(reader.keys())

    logging.info(f"Found {len(valid_ids)} valid samples across all data sources")

    # Build output JSON
    output = {"data_entry": triplet_info, "samples": sorted(valid_ids)}

    # Write to JSON
    logging.info(f"Writing dataset JSON to: {output_json}")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logging.info(f"Completed: {len(valid_ids)} valid samples written")


def get_parser():
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        description="Prepare dataset JSON from multiple data sources",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--triplets",
        type=str,
        nargs="+",
        required=True,
        help="List of name,path,reader triplets "
        "(e.g., audio1,/path/to/audio,lhotse_audio "
        "or text1,/path/to/text,text "
        "or dialogue1,/path/to/dialogue_folder,dialogue)",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="Logging level",
    )
    return parser


def main(cmd=None):
    """Run the main function."""
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    prepare_dataset_json(**kwargs)


if __name__ == "__main__":
    main()
