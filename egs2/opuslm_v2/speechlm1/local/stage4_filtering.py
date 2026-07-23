#!/usr/bin/env python3
"""Stage 4: Filtering with Gumbel top-k selection."""

import argparse
import os
from collections import defaultdict
from functools import partial
from multiprocessing import Pool

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_value(val, dtype=float):
    """Parse a value, returning None for 'N/A' or invalid values."""
    if val == "N/A":
        return None
    try:
        return dtype(val)
    except ValueError:
        return None


def load_samples(filepath, dataset):
    """Load samples from kept_utt_ids.txt file.

    Returns:
        List of tuples: ((dataset, utt_id), score, t2a_len, a2t_len)
        where t2a_len = text-to-audio length, a2t_len = audio-to-text length
    """
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 4:
                utt_id = parts[0]
                score = parse_value(parts[1], float)
                t2a_len = parse_value(parts[2], float)
                a2t_len = parse_value(parts[3], float)
                if score is not None and t2a_len is not None and a2t_len is not None:
                    samples.append(((dataset, utt_id), score, t2a_len, a2t_len))
    return samples


def compute_length_stats(samples, kept_keys=None):
    """Compute length statistics for samples.

    Args:
        samples: List of ((dataset, utt_id), score, t2a_len, a2t_len) tuples
        kept_keys: Optional set of (dataset, utt_id) to filter by.
                   If None, compute stats for all samples.

    Returns:
        Tuple of (per_dataset_stats, global_t2a, global_a2t, global_count)
        per_dataset_stats is a dict: {dataset: {"t2a": sum, "a2t": sum, "count": n}}
    """
    stats = defaultdict(lambda: {"t2a": 0.0, "a2t": 0.0, "count": 0})
    for (dataset, utt_id), score, t2a_len, a2t_len in samples:
        if kept_keys is None or (dataset, utt_id) in kept_keys:
            stats[dataset]["t2a"] += t2a_len
            stats[dataset]["a2t"] += a2t_len
            stats[dataset]["count"] += 1

    global_t2a = sum(s["t2a"] for s in stats.values())
    global_a2t = sum(s["a2t"] for s in stats.values())
    global_count = sum(s["count"] for s in stats.values())
    return stats, global_t2a, global_a2t, global_count


def gumbel_topk_filter(samples, keep_ratio, temperature, seed=42):
    """Apply Gumbel top-k trick to filter samples.

    Args:
        samples: List of ((dataset, utt_id), score, t2a_len, a2t_len) tuples
        keep_ratio: Fraction of samples to keep (e.g., 0.8 for 80%)
        temperature: Temperature for Gumbel noise (higher = more randomness)
        seed: Random seed for reproducibility

    Returns:
        Set of kept (dataset, utt_id) tuples
    """
    if not samples:
        return set()

    np.random.seed(seed)
    k = int(len(samples) * keep_ratio)

    # Add Gumbel noise: noisy_score = score + temperature * Gumbel(0, 1)
    # Gumbel(0,1) = -log(-log(U)) where U ~ Uniform(0,1)
    scores = np.array([s[1] for s in samples])
    u = np.random.uniform(0, 1, size=len(scores))
    gumbel_noise = -np.log(-np.log(u + 1e-10) + 1e-10)
    noisy_scores = scores + temperature * gumbel_noise

    # Get top-k indices
    topk_indices = np.argsort(noisy_scores)[-k:]

    return {samples[i][0] for i in topk_indices}


def plot_histogram_comparison(scores_before, scores_after, output_path, title):
    """Plot before/after histograms on the same plot."""
    if not scores_before:
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    min_score = min(scores_before)
    max_score = max(scores_before)
    bins = np.arange(min_score, max_score + 0.01, 0.01)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(scores_before, bins=bins, color="gray", alpha=0.5,
            label=f"Before (n={len(scores_before):,})", edgecolor="black")
    ax.hist(scores_after, bins=bins, color="steelblue", alpha=0.7,
            label=f"After (n={len(scores_after):,})", edgecolor="black")
    ax.set_xlabel("Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_dataset(dataset, input_dir):
    """Load samples from a single dataset."""
    input_file = os.path.join(input_dir, dataset, "kept_utt_ids.txt")

    if not os.path.exists(input_file):
        print(f"[{dataset}] File not found, skipping")
        return None

    samples = load_samples(input_file, dataset)
    print(f"[{dataset}] Loaded {len(samples)} items")
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=str, required=True)
    parser.add_argument("--audio_type", type=str, required=True,
                        choices=["speech", "music", "sound"])
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--discard_ratio", type=float, default=0.2,
                        help="Fraction of data to discard (default: 0.2)")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for Gumbel noise (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=32)
    args = parser.parse_args()

    datasets = args.datasets.split()
    print(f"Processing {len(datasets)} datasets for audio_type={args.audio_type}")
    print(f"Discard ratio: {args.discard_ratio}, Temperature: {args.temperature}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load all samples in parallel
    worker = partial(process_dataset, input_dir=args.input_dir)
    with Pool(args.num_workers) as pool:
        results = pool.map(worker, datasets)

    # Aggregate all samples globally
    all_samples = []
    for samples in results:
        if samples is not None:
            all_samples.extend(samples)

    print(f"\n{'=' * 60}")
    print(f"GLOBAL: {len(all_samples)} total items")

    # all_samples = all_samples[::100]

    # Compute length statistics before filtering
    stats_before, t2a_before, a2t_before, cnt_before = compute_length_stats(
        all_samples
    )

    # Apply Gumbel top-k filtering
    keep_ratio = 1.0 - args.discard_ratio
    kept_keys = gumbel_topk_filter(all_samples, keep_ratio, args.temperature,
                                   args.seed)
    print(f"After filtering: {len(kept_keys)} items "
          f"(kept {100 * keep_ratio:.0f}%)")

    # Compute length statistics after filtering
    stats_after, t2a_after, a2t_after, cnt_after = compute_length_stats(
        all_samples, kept_keys
    )

    # Print length statistics (token counts in Billion)
    print(f"\n{'=' * 100}")
    print("LENGTH STATISTICS (token counts in Billion)")
    print("=" * 100)
    header = (f"{'Dataset':<30} {'#Before':>10} {'#After':>10} "
              f"{'T2A Before':>12} {'T2A After':>12} "
              f"{'A2T Before':>12} {'A2T After':>12}")
    print(header)
    print("-" * 100)
    for dataset in sorted(stats_before.keys()):
        before = stats_before[dataset]
        after = stats_after.get(dataset, {"t2a": 0.0, "a2t": 0.0, "count": 0})
        print(f"{dataset:<30} {before['count']:>10,} {after['count']:>10,} "
              f"{before['t2a']/1e9:>12.4f} {after['t2a']/1e9:>12.4f} "
              f"{before['a2t']/1e9:>12.4f} {after['a2t']/1e9:>12.4f}")
    print("-" * 100)
    print(f"{'GLOBAL':<30} {cnt_before:>10,} {cnt_after:>10,} "
          f"{t2a_before/1e9:>12.4f} {t2a_after/1e9:>12.4f} "
          f"{a2t_before/1e9:>12.4f} {a2t_after/1e9:>12.4f}")
    print("=" * 100)

    # Collect scores before and after filtering
    scores_before = [s[1] for s in all_samples]
    scores_after = [s[1] for s in all_samples if s[0] in kept_keys]

    # Plot global histogram comparison
    global_dir = os.path.join(args.output_dir, "global")
    os.makedirs(global_dir, exist_ok=True)
    plot_histogram_comparison(
        scores_before, scores_after,
        os.path.join(global_dir, "global_score_hist.png"),
        f"Global - Score Distribution ({args.audio_type})"
    )
    print(f"Saved global histogram to {global_dir}/global_score_hist.png")

    # Group kept samples by dataset and write kept_utt_ids.txt
    kept_by_dataset = defaultdict(list)
    for (dataset, utt_id), score, t2a_len, a2t_len in all_samples:
        if (dataset, utt_id) in kept_keys:
            kept_by_dataset[dataset].append((utt_id, score))

    for dataset in datasets:
        dataset_dir = os.path.join(args.output_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        output_file = os.path.join(dataset_dir, "kept_utt_ids.txt")

        kept_items = kept_by_dataset.get(dataset, [])
        with open(output_file, "w", encoding="utf-8") as f:
            for utt_id, score in kept_items:
                f.write(f"{utt_id}\t{score:.6f}\n")

        print(f"[{dataset}] Saved {len(kept_items)} kept items to {output_file}")


if __name__ == "__main__":
    main()
