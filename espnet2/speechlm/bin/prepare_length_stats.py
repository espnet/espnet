#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Script for collecting sequence length statistics for efficient batching."""

import argparse
import json
import logging
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Tuple

import yaml

from espnet2.speechlm.dataloader.iterator import DataIteratorFactory
from espnet2.speechlm.model import _all_job_types


def get_parser() -> argparse.ArgumentParser:
    """Build argument parser for length statistics preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare Length Statistics for SpeechLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Configuration
    parser.add_argument(
        "--train-config",
        type=Path,
        required=True,
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exp/stats"),
        help="Directory to save length statistics",
    )

    # Data specifiers - simplified with helper function
    for split in ["train", "valid"]:
        for spec_type in ["unregistered", "registered"]:
            if spec_type == "unregistered":
                format_str = "task:name:data_json[:factor]"
                example = "asr:librispeech:train.json:2.0"
            else:
                format_str = "task:name[:factor]"
                example = "tts:ljspeech:1.5"

            parser.add_argument(
                f"--{split}-{spec_type}-specifier",
                type=str,
                default="",
                help=f"{spec_type.capitalize()} {split} data specifier. "
                f"Format: '{format_str}' (e.g., '{example}')",
            )

    # Processing options
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes for data loading",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    return parser


def worker(
    preprocessor,
    rank: int,
    world_size: int,
    unregistered_spec: str = "",
    registered_spec: str = "",
):
    """Worker function to collect length statistics for a data shard."""
    # Create iterator with appropriate specifier
    iterator = DataIteratorFactory(
        unregistered_specifier=unregistered_spec,
        registered_specifier=registered_spec,
        rank=rank,
        world_size=world_size,
        shuffle=False,
        sequential_load=True,
        num_workers=0,
        collate_fn=lambda x: x[0],
    ).get_iterator()

    # Collect statistics for this shard
    stats = {}
    for key, data_dict in iterator:
        stats[key] = preprocessor.find_length(key, data_dict)

    return stats


def collect_length_stats(
    preprocessor, num_workers: int, spec_type: str, specifier: str
) -> Dict[Tuple[str, ...], int]:
    """Collect length statistics from all worker processes.

    Args:
        preprocessor: Data preprocessor
        num_workers: Number of parallel workers
        spec_type: Either "unregistered" or "registered"
        specifier: The data specifier string

    Returns:
        Aggregated statistics dictionary
    """
    # Set up keyword arguments based on spec type
    kwargs = {
        "unregistered_spec": specifier if spec_type == "unregistered" else "",
        "registered_spec": specifier if spec_type == "registered" else "",
    }

    # Run workers in parallel
    with Pool(num_workers) as pool:
        results = [
            pool.apply_async(
                worker, args=(preprocessor, rank, num_workers), kwds=kwargs
            )
            for rank in range(num_workers)
        ]

        # Aggregate results from all workers
        aggregated_stats = {}
        for result in results:
            aggregated_stats.update(result.get())

    return aggregated_stats


def save_stats(stats: Dict[Tuple[str, ...], int], output_file: Path) -> None:
    """Save statistics dictionary to JSONL file."""
    logger = logging.getLogger(__name__)

    # Save to JSONL format
    with open(output_file, "w") as f:
        for key_tuple, value in stats.items():
            # Only keep the example_id, and discard the task and data_name
            json_obj = {key_tuple[2]: value}
            f.write(json.dumps(json_obj) + "\n")

    # Log summary
    if stats:
        total_frames = sum(stats.values())
        logger.info(
            f"Saved {len(stats)} entries to {output_file} | "
            f"Total: {total_frames} frames | "
            f"Avg: {total_frames / len(stats):.1f} frames/entry"
        )


def main():
    """Main entry point for length statistics preparation."""
    # Parse arguments
    args = get_parser().parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s (%(module)s:%(lineno)d) [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting length statistics preparation")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build preprocessor from config
    with open(args.train_config) as f:
        config = yaml.safe_load(f)

    job_template = _all_job_types[config["job_type"]](config)
    preprocessor = job_template.build_preprocessor()

    # Collect all specifiers to process
    specifiers = []
    for split in ["train", "valid"]:
        for spec_type in ["unregistered", "registered"]:
            spec_str = getattr(args, f"{split}_{spec_type}_specifier")
            if spec_str:
                for spec in spec_str.split():
                    specifiers.append((spec_type, spec))

    # Process each specifier
    for spec_type, specifier in specifiers:
        # Parse specifier and generate output filename
        parts = specifier.split(":")
        task, data_name = parts[0], parts[1]
        output_file = args.output_dir / f"stats_{task}_{data_name}.jsonl"

        # Skip if already exists
        if output_file.exists():
            logger.info(f"Skipping {specifier} - stats already exist")
            continue

        # Collect and save statistics
        logger.info(f"Processing {spec_type} specifier: {specifier}")
        stats = collect_length_stats(
            preprocessor, args.num_workers, spec_type, specifier
        )
        save_stats(stats, output_file)


if __name__ == "__main__":
    main()
