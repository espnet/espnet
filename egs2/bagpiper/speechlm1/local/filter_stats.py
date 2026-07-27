#!/usr/bin/env python3
"""Filter stats JSONL files based on kept utterance IDs."""

import argparse
import json
import os


def load_kept_utt_ids(filepath):
    """Load kept utterance IDs from stage4 output file."""
    kept_ids = set()
    if not os.path.exists(filepath):
        return kept_ids

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts:
                kept_ids.add(parts[0])
    return kept_ids


def filter_stats_file(input_path, output_path, kept_ids):
    """Filter a stats JSONL file to only include kept IDs."""
    if not os.path.exists(input_path):
        print(f"  Input file not found: {input_path}")
        return 0

    if not kept_ids:
        print(f"  No kept IDs, skipping")
        return 0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    count = 0
    with open(input_path, "r", encoding="utf-8") as fin:
        with open(output_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # Each line is {"utt_id": value}
                utt_id = list(obj.keys())[0]
                if utt_id in kept_ids:
                    fout.write(line + "\n")
                    count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--audio_type", type=str, required=True,
        choices=["music", "sound", "speech"]
    )
    parser.add_argument("--stats_root", type=str, required=True)
    parser.add_argument("--stage4_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    dataset = args.dataset

    # Load kept IDs
    kept_file = os.path.join(args.stage4_dir, dataset, "kept_utt_ids.txt")
    kept_ids = load_kept_utt_ids(kept_file)

    if not kept_ids:
        print(f"[{dataset}] No kept IDs found, skipping")
        return

    print(f"[{dataset}] Filtering stats with {len(kept_ids)} kept IDs")

    # Filter text_to_audio stats
    t2a_input = os.path.join(args.stats_root, f"stats_text_to_audio_{dataset}.jsonl")
    t2a_output = os.path.join(args.output_dir, f"stats_text_to_audio_{dataset}.jsonl")
    t2a_count = filter_stats_file(t2a_input, t2a_output, kept_ids)
    print(f"  text_to_audio: {t2a_count} entries -> {t2a_output}")

    # Filter audio_to_text stats
    a2t_input = os.path.join(args.stats_root, f"stats_audio_to_text_{dataset}.jsonl")
    a2t_output = os.path.join(args.output_dir, f"stats_audio_to_text_{dataset}.jsonl")
    a2t_count = filter_stats_file(a2t_input, a2t_output, kept_ids)
    print(f"  audio_to_text: {a2t_count} entries -> {a2t_output}")


if __name__ == "__main__":
    main()
