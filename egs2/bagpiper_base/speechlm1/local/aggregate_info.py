#!/usr/bin/env python3
"""
Stage 2: Aggregate information from multiple sources.
Drop data points that are missing from any source.
"""

import argparse
import glob
import json
import os
from functools import partial
from multiprocessing import Pool

import pandas as pd


def load_field_from_jsonl(filepath, field_names=None, nested_json_field=None):
    """Load utt_id and optionally field(s) from a single jsonl file.

    Args:
        filepath: Path to jsonl file
        field_names: None (utt_id only) or list of field names to extract
        nested_json_field: If set, parse this field as JSON first, then extract
    """
    results = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "utt_id" not in data:
                    continue
                utt_id = data["utt_id"]

                # Parse nested JSON if specified
                if nested_json_field and nested_json_field in data:
                    data = json.loads(data[nested_json_field])

                if field_names is None:
                    results[utt_id] = None
                else:
                    field_dict = {}
                    for fn in field_names:
                        if fn in data:
                            field_dict[fn] = data[fn]
                    if field_dict:
                        results[utt_id] = field_dict
            except json.JSONDecodeError:
                continue
    return results


def load_jsonl_dir(
    directory, field_names, num_workers, label, nested_json_field=None
):
    """Load data from all jsonl files in a directory."""
    jsonl_files = glob.glob(os.path.join(directory, "*.jsonl"))
    print(f"[{label}] Found {len(jsonl_files)} jsonl files")

    data = {}
    loader = partial(
        load_field_from_jsonl,
        field_names=field_names,
        nested_json_field=nested_json_field,
    )
    with Pool(num_workers) as pool:
        results = pool.map(loader, jsonl_files)
        for result in results:
            data.update(result)

    field_desc = f"with {field_names}" if field_names else ""
    print(f"[{label}] Loaded {len(data)} utt_ids {field_desc}")
    return data


def filter_and_update(utt_data, source_data, target_keys, label, exclude=False):
    """Filter utt_data and optionally add values from source_data.

    Args:
        utt_data: Global dict to update
        source_data: Source dict with utt_id -> value or utt_id -> {field: value}
        target_keys: None (filter only), list of keys to add, or "merge" to merge
        label: Label for logging
        exclude: If True, remove items IN source_data; otherwise keep only those
    """
    before_count = len(utt_data)

    if exclude:
        # Remove items that ARE in source_data
        utt_data = {k: v for k, v in utt_data.items() if k not in source_data}
    else:
        # Remove items that are NOT in source_data
        utt_data = {k: v for k, v in utt_data.items() if k in source_data}
        # Add values if target_keys is specified
        if target_keys == "merge":
            # Merge dict values directly
            for utt_id in utt_data:
                utt_data[utt_id].update(source_data[utt_id])
        elif target_keys:
            # Map source fields to target keys
            for utt_id in utt_data:
                src = source_data[utt_id]
                for tgt_key, src_key in target_keys.items():
                    if src_key in src:
                        utt_data[utt_id][tgt_key] = src[src_key]

    after_count = len(utt_data)
    print(f"[{label}] Removed {before_count - after_count} items")
    print(f"[{label}] Remaining {after_count} utt_ids")
    return utt_data


def write_chunk(args):
    """Write a chunk of data to a jsonl file.

    Args:
        args: Tuple of (output_path, chunk_data) where chunk_data is a list of
              (utt_id, data_dict) tuples.
    """
    output_path, chunk_data = args
    with open(output_path, "w", encoding="utf-8") as f:
        for utt_id, data in chunk_data:
            record = {"utt_id": utt_id, **data}
            f.write(json.dumps(record) + "\n")
    return output_path, len(chunk_data)


def load_heuristic_delete_ids(filepath):
    """Load delete ids from heuristic filtering result."""
    delete_ids = set()
    if not os.path.exists(filepath):
        print(f"[Heuristic] File not found: {filepath}")
        return delete_ids

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "utt_id" in data:
                    delete_ids.add(data["utt_id"])
            except json.JSONDecodeError:
                continue
    print(f"[Heuristic] Loaded {len(delete_ids)} delete ids")
    return delete_ids


def load_stats_jsonl(filepath, target_key, label):
    """Load stats from jsonl file with format {utt_id: value}.

    Args:
        filepath: Path to jsonl file
        target_key: Key name to store the value in output dict
        label: Label for logging
    """
    data = {}
    if not os.path.exists(filepath):
        print(f"[{label}] File not found: {filepath}")
        return data

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                # Format is {utt_id: value}
                for utt_id, value in row.items():
                    data[utt_id] = {target_key: value}
            except json.JSONDecodeError:
                continue
    print(f"[{label}] Loaded {len(data)} utt_ids with {target_key}")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate information from multiple sources"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Directory containing audio metadata.parquet",
    )
    parser.add_argument(
        "--rich_caption_dir",
        type=str,
        required=True,
        help="Directory containing rich caption jsonl files",
    )
    parser.add_argument(
        "--heuristic_delete_ids",
        type=str,
        required=True,
        help="Path to delete_ids.jsonl from heuristic filtering",
    )
    parser.add_argument(
        "--mos_dir",
        type=str,
        required=True,
        help="Directory containing MOS jsonl files",
    )
    parser.add_argument(
        "--clap_dir",
        type=str,
        required=True,
        help="Directory containing CLAP jsonl files",
    )
    parser.add_argument(
        "--aesthetics_dir",
        type=str,
        required=True,
        help="Directory containing aesthetics jsonl files",
    )
    parser.add_argument(
        "--llm_judge_dir",
        type=str,
        required=True,
        help="Directory containing LLM judge jsonl files",
    )
    parser.add_argument(
        "--stats_text_to_audio",
        type=str,
        required=True,
        help="Path to stats_text_to_audio jsonl file",
    )
    parser.add_argument(
        "--stats_audio_to_text",
        type=str,
        required=True,
        help="Path to stats_audio_to_text jsonl file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for aggregated results",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of workers for multiprocessing",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Check for done flag
    done_flag = os.path.join(args.output_dir, ".done")
    if os.path.exists(done_flag):
        print(f"[Skip] Done flag found at {done_flag}, skipping processing.")
        return

    # Global dict: utt_id -> {key1: value1, ..., keyN: valueN}
    utt_data = {}

    # === Source 1: Audio metadata ===
    audio_parquet = os.path.join(args.audio_dir, "metadata.parquet")
    wav_scp = os.path.join(args.audio_dir, "wav.scp")

    if os.path.exists(audio_parquet):
        audio_df = pd.read_parquet(audio_parquet)
        for _, row in audio_df.iterrows():
            utt_data[row["utt_id"]] = {}
        print(f"[Audio] Loaded {len(utt_data)} utt_ids from metadata.parquet")
    elif os.path.exists(wav_scp):
        with open(wav_scp, "r") as f:
            for line in f:
                utt_id = line.strip().split()[0]
                utt_data[utt_id] = {}
        print(f"[Audio] Loaded {len(utt_data)} utt_ids from wav.scp")
    else:
        raise FileNotFoundError(
            f"Audio metadata not found: neither {audio_parquet} nor {wav_scp} exists"
        )

    # === Source 2: Rich caption (filter only) ===
    rich_caption_data = load_jsonl_dir(
        args.rich_caption_dir, None, args.num_workers, "Rich Caption"
    )
    utt_data = filter_and_update(
        utt_data, rich_caption_data, None, "Rich Caption"
    )

    # === Source 3: Heuristic filtering (delete list) ===
    delete_ids = load_heuristic_delete_ids(args.heuristic_delete_ids)
    utt_data = filter_and_update(
        utt_data, delete_ids, None, "Heuristic", exclude=True
    )

    # === Source 4: MOS scores ===
    mos_data = load_jsonl_dir(args.mos_dir, ["MEAN"], args.num_workers, "MOS")
    utt_data = filter_and_update(
        utt_data, mos_data, {"mos_mean": "MEAN"}, "MOS"
    )

    # === Source 5: CLAP scores ===
    clap_data = load_jsonl_dir(
        args.clap_dir, ["cosine_similarity"], args.num_workers, "CLAP"
    )
    utt_data = filter_and_update(
        utt_data, clap_data, {"clap_similarity": "cosine_similarity"}, "CLAP"
    )

    # === Source 6: Aesthetics scores ===
    aesthetics_data = load_jsonl_dir(
        args.aesthetics_dir, ["MEAN"], args.num_workers, "Aesthetics"
    )
    utt_data = filter_and_update(
        utt_data, aesthetics_data, {"aesthetics_mean": "MEAN"}, "Aesthetics"
    )

    # === Source 7: LLM judge ===
    llm_judge_fields = [
        "audio_type",
        "is_pure_english",
        "is_audio_focused",
        "intelligibility_score",
        "complexity_score",
        "diversity_score",
    ]
    llm_judge_data = load_jsonl_dir(
        args.llm_judge_dir,
        llm_judge_fields,
        args.num_workers,
        "LLM Judge",
        nested_json_field="response",
    )
    utt_data = filter_and_update(
        utt_data, llm_judge_data, "merge", "LLM Judge"
    )

    # === Source 8: Stats text_to_audio ===
    stats_t2a_data = load_stats_jsonl(
        args.stats_text_to_audio, "text_to_audio_len", "Stats T2A"
    )
    utt_data = filter_and_update(
        utt_data, stats_t2a_data, "merge", "Stats T2A"
    )

    # === Source 9: Stats audio_to_text ===
    stats_a2t_data = load_stats_jsonl(
        args.stats_audio_to_text, "audio_to_text_len", "Stats A2T"
    )
    utt_data = filter_and_update(
        utt_data, stats_a2t_data, "merge", "Stats A2T"
    )

    # === Export results ===
    print(f"[Export] Writing {len(utt_data)} items to {args.output_dir}")
    lines_per_file = 1_000_000
    utt_ids = list(utt_data.keys())

    # Prepare chunks for parallel writing
    write_tasks = []
    for file_idx, start_idx in enumerate(range(0, len(utt_ids), lines_per_file)):
        end_idx = min(start_idx + lines_per_file, len(utt_ids))
        output_path = os.path.join(args.output_dir, f"part_{file_idx:04d}.jsonl")
        chunk_data = [
            (utt_id, utt_data[utt_id]) for utt_id in utt_ids[start_idx:end_idx]
        ]
        write_tasks.append((output_path, chunk_data))

    # Write chunks in parallel
    with Pool(args.num_workers) as pool:
        results = pool.map(write_chunk, write_tasks)

    for output_path, count in results:
        print(f"[Export] Wrote {count} items to {output_path}")

    print(f"[Export] Done. Total {len(write_tasks)} files written.")

    # Write done flag
    with open(done_flag, "w") as f:
        f.write("done\n")
    print(f"[Export] Done flag written to {done_flag}")


if __name__ == "__main__":
    main()
