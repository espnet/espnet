#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Stage 6: Build curated datasets from filtered utterance IDs.

This script processes all datasets for all audio types:
1. Filter data entries (parquet/wav.scp) based on kept_utt_ids from stage4
2. Build dataset.json directly
3. Filter stats JSONL files
4. Generate registry YAML for each audio type

Usage:
    python3 local/stage5_build_curated_dataset.py \
        --datasets "audiocaps audioset ..." \
        --audio_types "speech sound music" \
        --registry_file /path/to/registry.yaml \
        --stage4_root /path/to/data_curation \
        --stats_root /path/to/stats \
        --output_root /path/to/output \
        --version v1 \
        --num_workers 8
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml


def load_kept_utt_ids(filepath):
    """Load kept utterance IDs from stage4 output file."""
    kept_ids = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts:
                kept_ids.add(parts[0])
    return kept_ids


def filter_parquet(input_path, output_path, kept_ids):
    """Filter a parquet file to only include kept IDs."""
    # Disable PyArrow threading to prevent deadlocks
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)

    table = pq.read_table(input_path)
    mask = pc.is_in(table.column("utt_id"), value_set=pa.array(list(kept_ids)))
    filtered_table = table.filter(mask)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pq.write_table(filtered_table, output_path)

    return len(filtered_table), set(filtered_table.column("utt_id").to_pylist())


def filter_wavscp(input_path, output_path, kept_ids):
    """Filter a wav.scp-style file to only include kept IDs."""
    count = 0
    actual_ids = set()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as fin:
        with open(output_path, "w", encoding="utf-8") as fout:
            for line in fin:
                parts = line.strip().split(maxsplit=1)
                if parts and parts[0] in kept_ids:
                    fout.write(line)
                    actual_ids.add(parts[0])
                    count += 1
    return count, actual_ids


def filter_stats_file(input_path, output_path, kept_ids):
    """Filter a stats JSONL file to only include kept IDs."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    count = 0
    with open(input_path, "r", encoding="utf-8") as fin:
        with open(output_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if list(obj.keys())[0] in kept_ids:
                    fout.write(line + "\n")
                    count += 1
    return count


def process_dataset(dataset, audio_type, registry, stage4_root, stats_root,
                    output_root, version):
    """Process a single dataset for a given audio type."""
    stage4_dir = os.path.join(stage4_root, f"stage4_filtering_{audio_type}_{version}")
    output_dir = os.path.join(
        output_root, f"stage5_curated_{audio_type}_{version}"
    )

    # Check if dataset exists in registry
    assert dataset in registry, f"Dataset '{dataset}' not found in registry"

    # Load kept IDs
    kept_file = os.path.join(stage4_dir, dataset, "kept_utt_ids.txt")
    assert os.path.exists(kept_file), f"Kept IDs file not found: {kept_file}"

    kept_ids = load_kept_utt_ids(kept_file)

    # Load original dataset JSON
    orig_json_path = registry[dataset]["path"]
    with open(orig_json_path, "r", encoding="utf-8") as f:
        orig_data = json.load(f)

    # Filter each data entry
    new_entries = []
    all_actual_ids = None

    for entry in orig_data["data_entry"]:
        name = entry["name"]
        orig_path = entry["path"]
        reader = entry["reader"]
        output_path = os.path.join(
            output_dir, dataset, name, os.path.basename(orig_path)
        )

        if orig_path.endswith(".parquet"):
            count, actual_ids = filter_parquet(orig_path, output_path, kept_ids)
            print('done filtering parquet: ', orig_path)
        else:
            count, actual_ids = filter_wavscp(orig_path, output_path, kept_ids)

        assert count > 0, f"No samples after filtering {orig_path}"

        new_entries.append({
            "name": name,
            "path": str(Path(output_path).resolve()),
            "reader": reader
        })

        all_actual_ids = (
            actual_ids if all_actual_ids is None
            else all_actual_ids & actual_ids
        )

    assert all_actual_ids, "No common samples across entries"

    # Build dataset JSON
    dataset_json_path = os.path.join(output_dir, dataset, "dataset.json")
    os.makedirs(os.path.dirname(dataset_json_path), exist_ok=True)
    with open(dataset_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"data_entry": new_entries, "samples": sorted(all_actual_ids)},
            f, indent=2, ensure_ascii=False
        )

    # Filter stats files (output to same dir with audio_type and version in name)
    for task in ["text_to_audio", "audio_to_text"]:
        inp = os.path.join(stats_root, f"stats_{task}_{dataset}.jsonl")
        out = os.path.join(
            stats_root, f"stats_{task}_{dataset}_{audio_type}_{version}.jsonl"
        )
        if os.path.exists(inp):
            filter_stats_file(inp, out, kept_ids)

    return {
        "dataset": dataset,
        "audio_type": audio_type,
        "num_samples": len(all_actual_ids),
        "dataset_json": dataset_json_path,
    }


def generate_registry(output_dir, datasets, audio_type, version):
    """Generate registry YAML for an audio type."""
    registry_path = os.path.join(output_dir, "registry.yaml")
    registry_content = {}

    for dataset in datasets:
        dataset_json = os.path.join(output_dir, dataset, "dataset.json")
        if os.path.exists(dataset_json):
            registry_content[f"{dataset}_{audio_type}_{version}"] = {
                "path": dataset_json
            }

    with open(registry_path, "w", encoding="utf-8") as f:
        yaml.dump(registry_content, f, default_flow_style=False, sort_keys=False)

    return registry_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", type=str, required=True,
        help="Space-separated list of dataset names"
    )
    parser.add_argument(
        "--audio_types", type=str, default="speech sound music",
        help="Space-separated list of audio types"
    )
    parser.add_argument("--registry_file", type=str, required=True)
    parser.add_argument(
        "--stage4_root", type=str, required=True,
        help="Root directory containing stage4_filtering_* directories"
    )
    parser.add_argument("--stats_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    datasets = args.datasets.split()
    audio_types = args.audio_types.split()

    # Load registry
    with open(args.registry_file, "r", encoding="utf-8") as f:
        registry = yaml.safe_load(f)

    print(f"Processing {len(datasets)} datasets for audio types: {audio_types}")
    print(f"Using {args.num_workers} workers")

    # Process all audio types
    for audio_type in audio_types:
        print(f"\n{'='*60}")
        print(f"Processing audio_type={audio_type}")
        print(f"{'='*60}")

        output_dir = os.path.join(
            args.output_root, f"stage5_curated_{audio_type}_{args.version}"
        )
        os.makedirs(output_dir, exist_ok=True)

        results = []
        # Process datasets in parallel
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(
                    process_dataset,
                    dataset,
                    audio_type,
                    registry,
                    args.stage4_root,
                    args.stats_root,
                    args.output_root,
                    args.version
                ): dataset
                for dataset in datasets
            }

            for future in as_completed(futures):
                dataset = futures[future]
                print('waiting for : ', future, flush=True  )
                result = future.result()
                results.append(result)
                print(f"  [OK] {dataset}: {result['num_samples']} samples")

        # Generate registry
        registry_path = generate_registry(output_dir, datasets, audio_type, args.version)
        print(f"Created registry: {registry_path}")

        # Summary
        total_samples = sum(r["num_samples"] for r in results)
        print(f"Summary: {len(results)}/{len(datasets)} datasets succeeded")
        print(f"Total samples: {total_samples}")

    print("\nAll audio types processed successfully!")


if __name__ == "__main__":
    main()