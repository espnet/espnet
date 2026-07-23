#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Compute length statistics for curated datasets.

This script computes:
1. For each dataset and each audio type: number of samples and total tokens
2. For each audio type overall: number of samples and total tokens
3. Token counts in B (billions), sample counts in M (millions)

Usage:
    python3 local/compute_length_stats.py \
        --datasets "audiocaps audioset ..." \
        --audio_types "music sound speech" \
        --stats_root /path/to/stats \
        --version v1 \
        --num_workers 32
"""

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_stats_file(filepath):
    """Process a single stats JSONL file and return (num_samples, total_tokens).

    Each line in the file has format: {"utt_id": token_count}
    """
    if not os.path.exists(filepath):
        return 0, 0

    num_samples = 0
    total_tokens = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            num_samples += 1
            total_tokens += sum(data.values())

    return num_samples, total_tokens


def process_dataset_audio_type(args_tuple):
    """Process a single dataset-audio_type combination."""
    dataset, audio_type, stats_root, version = args_tuple

    # Check both text_to_audio and audio_to_text stats
    results = {}
    for task in ["text_to_audio", "audio_to_text"]:
        filepath = os.path.join(
            stats_root, f"stats_{task}_{dataset}_{audio_type}_{version}.jsonl"
        )
        num_samples, total_tokens = process_stats_file(filepath)
        results[task] = {"num_samples": num_samples, "total_tokens": total_tokens}

    return dataset, audio_type, results


def format_billions(value):
    """Format value in billions."""
    return f"{value / 1e9:.4f}B"


def format_millions(value):
    """Format value in millions."""
    return f"{value / 1e6:.4f}M"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", type=str, required=True,
        help="Space-separated list of dataset names"
    )
    parser.add_argument(
        "--audio_types", type=str, default="music sound speech",
        help="Space-separated list of audio types"
    )
    parser.add_argument("--stats_root", type=str, required=True)
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--num_workers", type=int, default=32)
    args = parser.parse_args()

    datasets = args.datasets.split()
    audio_types = args.audio_types.split()

    print(f"Computing stats for {len(datasets)} datasets")
    print(f"Audio types: {audio_types}")
    print(f"Stats root: {args.stats_root}")
    print(f"Version: {args.version}")
    print()

    # Prepare all tasks
    tasks = [
        (dataset, audio_type, args.stats_root, args.version)
        for dataset in datasets
        for audio_type in audio_types
    ]

    # Process in parallel
    all_results = {}
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(process_dataset_audio_type, task): task
            for task in tasks
        }
        for future in as_completed(futures):
            dataset, audio_type, results = future.result()
            if dataset not in all_results:
                all_results[dataset] = {}
            all_results[dataset][audio_type] = results

    # Aggregate by audio type
    audio_type_totals = {
        at: {"text_to_audio": {"num_samples": 0, "total_tokens": 0},
             "audio_to_text": {"num_samples": 0, "total_tokens": 0}}
        for at in audio_types
    }

    # Print per-dataset, per-audio_type statistics
    print("=" * 100)
    print("Per-Dataset, Per-Audio-Type Statistics")
    print("=" * 100)

    header = f"{'Dataset':<30} {'Audio Type':<12} {'Task':<15} "
    header += f"{'Samples':<12} {'Tokens':<15}"
    print(header)
    print("-" * 100)

    for dataset in datasets:
        if dataset not in all_results:
            continue
        for audio_type in audio_types:
            if audio_type not in all_results[dataset]:
                continue
            for task in ["text_to_audio", "audio_to_text"]:
                stats = all_results[dataset][audio_type][task]
                if stats["num_samples"] == 0:
                    continue

                samples_str = format_millions(stats["num_samples"])
                tokens_str = format_billions(stats["total_tokens"])

                print(
                    f"{dataset:<30} {audio_type:<12} {task:<15} "
                    f"{samples_str:<12} {tokens_str:<15}"
                )

                # Add to totals
                audio_type_totals[audio_type][task]["num_samples"] += (
                    stats["num_samples"]
                )
                audio_type_totals[audio_type][task]["total_tokens"] += (
                    stats["total_tokens"]
                )

    # Print per-audio_type totals
    print()
    print("=" * 100)
    print("Per-Audio-Type Overall Statistics")
    print("=" * 100)

    header = f"{'Audio Type':<15} {'Task':<15} {'Samples':<12} {'Tokens':<15}"
    print(header)
    print("-" * 100)

    grand_total_samples = 0
    grand_total_tokens = 0

    for audio_type in audio_types:
        for task in ["text_to_audio", "audio_to_text"]:
            stats = audio_type_totals[audio_type][task]
            if stats["num_samples"] == 0:
                continue

            samples_str = format_millions(stats["num_samples"])
            tokens_str = format_billions(stats["total_tokens"])

            print(
                f"{audio_type:<15} {task:<15} {samples_str:<12} {tokens_str:<15}"
            )

            grand_total_samples += stats["num_samples"]
            grand_total_tokens += stats["total_tokens"]

    # Print grand totals
    print()
    print("=" * 100)
    print("Grand Total")
    print("=" * 100)
    print(f"Total Samples: {format_millions(grand_total_samples)}")
    print(f"Total Tokens:  {format_billions(grand_total_tokens)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
