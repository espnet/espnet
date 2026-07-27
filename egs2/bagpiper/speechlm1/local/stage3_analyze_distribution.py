#!/usr/bin/env python3
"""
Stage 3: Analyze distributions of aggregated metrics.
For each metric, either filter by a specified value or count the distribution.
"""

import argparse
import glob
import json
import math
import os
from collections import Counter
from functools import partial
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server
import matplotlib.pyplot as plt


# Define metric types
FLOAT_METRICS = ["mos_mean", "clap_similarity", "aesthetics_mean"]
INTEGER_METRICS = [
    "intelligibility_score",
    "complexity_score",
    "diversity_score",
    "text_to_audio_len",
    "audio_to_text_len",
]
CATEGORICAL_METRICS = ["audio_type", "is_pure_english", "is_audio_focused"]

# Audio type categories
AUDIO_TYPE_CHOICES = ["speech", "music", "sound_effects"]

# Scale all metrics to [0, 1] for weighted scoring.
# Aspects like audio-only, text-only and text-audio should have equal weights.
METRIC_WEIGHTS = {
    "intelligibility_score": 0.2 * 1/3,
    "complexity_score": 0.2 * 1/3,
    "diversity_score": 0.2 * 1/3,
    "mos_mean": 0.2,
    "clap_similarity": 1.0,
    "aesthetics_mean": 0.1,
}


def calculate_overall_score(data, filters):
    """Calculate overall score as weighted sum of non-filtered metrics.

    Args:
        data: Dict containing metric values
        filters: Dict of metrics used for filtering (to exclude from score)

    Returns:
        Float score, or None if no valid metrics
    """
    audio_type = data.get("audio_type")
    score = 0.0

    for metric, weight in METRIC_WEIGHTS.items():
        # Skip metrics used for filtering
        if metric in filters:
            continue
        # For speech: don't consider aesthetics_mean
        if audio_type == "speech" and metric == "aesthetics_mean":
            continue
        # For non-speech: don't consider mos_mean
        if audio_type != "speech" and metric == "mos_mean":
            continue

        value = data.get(metric)
        if value is None:
            continue

        score += weight * value

    return score if score > 0 else None


def load_minhush_blacklist(minhush_root, dataset):
    """Load blacklist utt_ids from minhush delete file.

    Args:
        minhush_root: Root directory containing minhush delete files
        dataset: Dataset name to load blacklist for

    Returns:
        Set of utt_ids to filter out
    """
    blacklist = set()
    delete_file = os.path.join(minhush_root, f"{dataset}_delete.jsonl")

    if not os.path.exists(delete_file):
        print(f"  Warning: Minhush delete file not found: {delete_file}")
        return blacklist

    with open(delete_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                utt_id = data.get("utt_id")
                if utt_id:
                    blacklist.add(utt_id)
            except json.JSONDecodeError:
                continue

    print(f"  Loaded {len(blacklist)} blacklisted utt_ids from minhush")
    return blacklist


def float_to_bucket(value, increment=0.01):
    """Convert a float value to a bucket key.

    Args:
        value: Float value to bucket
        increment: Bucket size (default 0.1)

    Returns:
        Bucket key as a string like "0.0-0.1", "0.1-0.2", etc.
    """
    if value is None or not isinstance(value, (int, float)):
        return "invalid"
    bucket_idx = int(math.floor(value / increment))
    lower = bucket_idx * increment
    upper = lower + increment
    return f"{lower:.1f}-{upper:.1f}"


def process_jsonl_file(filepath, filters, metrics_to_count, blacklist=None):
    """Load, filter, and count distributions from a single jsonl file.

    Args:
        filepath: Path to jsonl file
        filters: Dict of metric -> value for filtering
        metrics_to_count: List of metrics to count distributions for
        blacklist: Set of utt_ids to filter out (optional)

    Returns:
        Dict with 'total', 'filtered', 'counters', 'kept_items'
    """
    if blacklist is None:
        blacklist = set()
    counters = {metric: Counter() for metric in metrics_to_count}
    total_count = 0
    filtered_count = 0
    # Each item: (utt_id, score, text_to_audio_len, audio_to_text_len)
    kept_items = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                total_count += 1

                # Check blacklist first
                utt_id = data.get("utt_id")
                if utt_id in blacklist:
                    continue

                # Apply filters
                skip = False
                for metric, threshold in filters.items():
                    data_value = data.get(metric)
                    if metric in FLOAT_METRICS:
                        # Float: keep if value > threshold (strictly above)
                        if data_value is None or data_value <= threshold:
                            skip = True
                            break
                    elif metric in INTEGER_METRICS:
                        # Integer: keep if value >= threshold
                        if data_value is None or data_value < threshold:
                            skip = True
                            break
                    elif metric in CATEGORICAL_METRICS:
                        # Categorical: keep if exact match
                        if data_value != threshold:
                            skip = True
                            break

                if skip:
                    continue

                filtered_count += 1

                # Calculate overall score and collect item info
                overall_score = calculate_overall_score(data, filters)
                text_to_audio_len = data.get("text_to_audio_len")
                audio_to_text_len = data.get("audio_to_text_len")
                kept_items.append((
                    utt_id, overall_score, text_to_audio_len, audio_to_text_len
                ))

                # Count distributions for non-filtered metrics
                for metric in metrics_to_count:
                    val = data.get(metric)
                    if val is None:
                        counters[metric]["missing"] += 1
                    elif metric in FLOAT_METRICS:
                        bucket = float_to_bucket(val)
                        counters[metric][bucket] += 1
                    else:
                        counters[metric][val] += 1

            except json.JSONDecodeError:
                continue

    return {
        "total": total_count,
        "filtered": filtered_count,
        "counters": counters,
        "kept_items": kept_items,
    }


def load_and_analyze_dataset(
    input_dir, num_workers, filters, metrics_to_count, blacklist=None
):
    """Load and analyze all jsonl files from a dataset directory.

    Args:
        input_dir: Directory containing jsonl files
        num_workers: Number of workers for multiprocessing
        filters: Dict of metric -> value for filtering
        metrics_to_count: List of metrics to count distributions for
        blacklist: Set of utt_ids to filter out (optional)

    Returns:
        Tuple of (total, filtered, aggregated_counters, kept_items)
    """
    if blacklist is None:
        blacklist = set()
    jsonl_files = glob.glob(os.path.join(input_dir, "*.jsonl"))
    if not jsonl_files:
        print(f"  No jsonl files found in {input_dir}")
        return 0, 0, {}, []

    print(f"  Found {len(jsonl_files)} jsonl files")

    worker = partial(
        process_jsonl_file,
        filters=filters,
        metrics_to_count=metrics_to_count,
        blacklist=blacklist,
    )

    with Pool(num_workers) as pool:
        results = pool.map(worker, jsonl_files)

    # Aggregate results from all workers
    total_count = 0
    filtered_count = 0
    aggregated_counters = {metric: Counter() for metric in metrics_to_count}
    kept_items = []

    for result in results:
        total_count += result["total"]
        filtered_count += result["filtered"]
        for metric, counter in result["counters"].items():
            aggregated_counters[metric].update(counter)
        kept_items.extend(result["kept_items"])

    return total_count, filtered_count, aggregated_counters, kept_items


def plot_histogram(args):
    """Plot and save a histogram for a metric distribution.

    Args:
        args: Tuple of (metric, counter, output_path, title_prefix)

    Returns:
        output_path if successful, None otherwise
    """
    metric, counter, output_path, title_prefix = args

    if not counter:
        return None

    total = sum(counter.values())
    if total == 0:
        return None

    # Prepare data for plotting
    if metric in FLOAT_METRICS or metric == "overall_score":
        # Sort float buckets numerically
        def sort_key(item):
            key = item[0]
            if key in ("invalid", "missing", "N/A"):
                return (1, 0, key)
            try:
                return (0, float(key.split("-")[0]), key)
            except (ValueError, IndexError):
                return (1, 0, key)

        sorted_items = sorted(counter.items(), key=sort_key)
    elif metric in INTEGER_METRICS:
        # Sort integer keys numerically
        def sort_key(item):
            key = item[0]
            if key in ("invalid", "missing", "N/A"):
                return (1, 0)
            try:
                return (0, int(key))
            except (ValueError, TypeError):
                return (1, 0)

        sorted_items = sorted(counter.items(), key=sort_key)
    else:
        # Categorical: sort by count descending
        sorted_items = counter.most_common()

    labels = [str(k) for k, _ in sorted_items]
    counts = [c for _, c in sorted_items]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot bars
    x_pos = range(len(labels))
    bars = ax.bar(x_pos, counts, color="steelblue", edgecolor="black")

    # Configure axes
    ax.set_xlabel(metric, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"{title_prefix} - {metric} (n={total:,})", fontsize=14)

    # Set x-axis labels
    if len(labels) <= 30:
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    else:
        # Too many labels, show every nth
        step = max(1, len(labels) // 20)
        ax.set_xticks(x_pos[::step])
        ax.set_xticklabels(labels[::step], rotation=45, ha="right", fontsize=8)

    # Add percentage labels on top of bars (only for bars with > 1% of total)
    for bar, count in zip(bars, counts):
        pct = 100.0 * count / total
        if pct >= 1.0:
            ax.annotate(
                f"{pct:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_histograms_parallel(plot_tasks, num_workers):
    """Plot multiple histograms in parallel.

    Args:
        plot_tasks: List of (metric, counter, output_path, title_prefix) tuples
        num_workers: Number of workers for multiprocessing

    Returns:
        List of successfully saved output paths
    """
    if not plot_tasks:
        return []

    with Pool(num_workers) as pool:
        results = pool.map(plot_histogram, plot_tasks)

    return [r for r in results if r is not None]


def main():
    parser = argparse.ArgumentParser(
        description="Analyze distributions of aggregated metrics"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="Space-separated list of dataset names",
    )
    parser.add_argument(
        "--input_base_dir",
        type=str,
        required=True,
        help="Base directory containing dataset subdirectories with jsonl files",
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default=None,
        help="Output base directory for distribution results (optional)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of workers for multiprocessing",
    )

    # Minhush blacklist filtering
    parser.add_argument(
        "--minhush_root",
        type=str,
        default=None,
        help="Root directory containing minhush delete files",
    )
    parser.add_argument(
        "--use_minhush",
        action="store_true",
        help="Enable minhush blacklist filtering",
    )

    # Float metric filters (keep if > value)
    parser.add_argument(
        "--mos_mean",
        type=float,
        default=None,
        help="Filter: keep if mos_mean > this value",
    )
    parser.add_argument(
        "--clap_similarity",
        type=float,
        default=None,
        help="Filter: keep if clap_similarity > this value",
    )
    parser.add_argument(
        "--aesthetics_mean",
        type=float,
        default=None,
        help="Filter: keep if aesthetics_mean > this value",
    )

    # Integer metric filters (keep if >= value)
    parser.add_argument(
        "--intelligibility_score",
        type=int,
        default=None,
        help="Filter: keep if intelligibility_score >= this value",
    )
    parser.add_argument(
        "--complexity_score",
        type=int,
        default=None,
        help="Filter: keep if complexity_score >= this value",
    )
    parser.add_argument(
        "--diversity_score",
        type=int,
        default=None,
        help="Filter: keep if diversity_score >= this value",
    )
    parser.add_argument(
        "--text_to_audio_len",
        type=int,
        default=None,
        help="Filter: keep if text_to_audio_len >= this value",
    )
    parser.add_argument(
        "--audio_to_text_len",
        type=int,
        default=None,
        help="Filter: keep if audio_to_text_len >= this value",
    )

    # Categorical metric filters (keep if equals)
    parser.add_argument(
        "--audio_type",
        type=str,
        choices=AUDIO_TYPE_CHOICES,
        default=None,
        help="Filter: keep if audio_type equals this value",
    )
    parser.add_argument(
        "--is_pure_english",
        type=str,
        choices=["true", "false"],
        default=None,
        help="Filter: keep if is_pure_english equals this value",
    )
    parser.add_argument(
        "--is_audio_focused",
        type=str,
        choices=["true", "false"],
        default=None,
        help="Filter: keep if is_audio_focused equals this value",
    )

    args = parser.parse_args()

    # Parse datasets
    datasets = args.datasets.split()
    print(f"Processing {len(datasets)} datasets: {datasets}")

    # Build filters dict
    filters = {}
    for metric in FLOAT_METRICS:
        value = getattr(args, metric, None)
        if value is not None:
            filters[metric] = value

    for metric in INTEGER_METRICS:
        value = getattr(args, metric, None)
        if value is not None:
            filters[metric] = value

    # Handle categorical args (convert string to proper type)
    if args.audio_type is not None:
        filters["audio_type"] = args.audio_type
    if args.is_pure_english is not None:
        filters["is_pure_english"] = args.is_pure_english.lower() == "true"
    if args.is_audio_focused is not None:
        filters["is_audio_focused"] = args.is_audio_focused.lower() == "true"

    # Determine which metrics to count (those not used for filtering)
    metrics_to_count = [
        m for m in FLOAT_METRICS + INTEGER_METRICS + CATEGORICAL_METRICS
        if m not in filters
    ]

    print(f"Filters: {filters}")
    print(f"Metrics to count: {metrics_to_count}")

    # Global aggregated counters
    global_counters = {metric: Counter() for metric in metrics_to_count}
    global_score_counter = Counter()
    global_total = 0
    global_filtered = 0

    # Per-dataset results
    dataset_results = {}

    # Process each dataset one by one
    for dataset in datasets:
        input_dir = os.path.join(args.input_base_dir, dataset)
        print(f"\n[{dataset}] Loading from {input_dir}")

        if not os.path.exists(input_dir):
            print(f"  Directory not found, skipping")
            continue

        # Load minhush blacklist if enabled
        blacklist = set()
        if args.use_minhush and args.minhush_root:
            blacklist = load_minhush_blacklist(args.minhush_root, dataset)

        # Load and analyze data with multiprocessing
        total, filtered, counters, kept_items = load_and_analyze_dataset(
            input_dir,
            args.num_workers,
            filters,
            metrics_to_count,
            blacklist=blacklist,
        )

        print(f"  Total: {total}, After filtering: {filtered}")
        global_total += total
        global_filtered += filtered

        # Store per-dataset results
        dataset_results[dataset] = {
            "total": total,
            "filtered": filtered,
            "counters": counters,
            "kept_items": kept_items,
        }

        # Merge counters into global
        for metric, counter in counters.items():
            global_counters[metric].update(counter)

        # Collect scores for global histogram
        for _, score, _, _ in kept_items:
            bucket = float_to_bucket(score) if score is not None else "N/A"
            global_score_counter[bucket] += 1

    # Print global summary
    print("\n" + "=" * 60)
    print("GLOBAL DISTRIBUTION SUMMARY")
    print(f"Total items loaded: {global_total}")
    print(f"Total items after filtering: {global_filtered}")
    print("=" * 60)

    # Save results if output_base_dir specified
    if args.output_base_dir:
        os.makedirs(args.output_base_dir, exist_ok=True)

        # Collect all plot tasks for parallel plotting
        plot_tasks = []

        # Save per-dataset results and prepare plot tasks
        for dataset, results in dataset_results.items():
            dataset_dir = os.path.join(args.output_base_dir, dataset)
            os.makedirs(dataset_dir, exist_ok=True)

            # Save dataset distributions as JSON
            output_file = os.path.join(dataset_dir, "distributions.json")
            output_data = {
                "total_items": results["total"],
                "total_filtered": results["filtered"],
                "filters": {k: str(v) for k, v in filters.items()},
                "distributions": {
                    metric: dict(counter)
                    for metric, counter in results["counters"].items()
                },
            }
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2)

            # Save dataset kept utt_ids with scores and lengths
            # Format: utt_id<TAB>score<TAB>text_to_audio_len<TAB>audio_to_text_len
            utt_ids_file = os.path.join(dataset_dir, "kept_utt_ids.txt")
            with open(utt_ids_file, "w", encoding="utf-8") as f:
                for utt_id, score, t2a_len, a2t_len in results["kept_items"]:
                    score_str = f"{score:.6f}" if score is not None else "N/A"
                    t2a_str = str(t2a_len) if t2a_len is not None else "N/A"
                    a2t_str = str(a2t_len) if a2t_len is not None else "N/A"
                    f.write(f"{utt_id}\t{score_str}\t{t2a_str}\t{a2t_str}\n")

            # Add plot tasks for this dataset
            for metric in metrics_to_count:
                if results["counters"][metric]:
                    plot_path = os.path.join(
                        dataset_dir, f"{dataset}_{metric}.png"
                    )
                    plot_tasks.append((
                        metric,
                        results["counters"][metric],
                        plot_path,
                        dataset,
                    ))

            print(f"\n[{dataset}] Saved to {dataset_dir}")
            print(f"  - distributions.json")
            print(f"  - kept_utt_ids.txt ({len(results['kept_items'])} items)")

        # Save global aggregated results in a subfolder
        global_dir = os.path.join(args.output_base_dir, "global")
        os.makedirs(global_dir, exist_ok=True)

        global_file = os.path.join(global_dir, "distributions.json")
        global_output_data = {
            "total_items": global_total,
            "total_filtered": global_filtered,
            "filters": {k: str(v) for k, v in filters.items()},
            "distributions": {
                metric: dict(counter)
                for metric, counter in global_counters.items()
            },
        }
        with open(global_file, "w", encoding="utf-8") as f:
            json.dump(global_output_data, f, indent=2)
        print(f"\nSaved global distributions to {global_dir}")

        # Add plot tasks for global distributions
        for metric in metrics_to_count:
            if global_counters[metric]:
                plot_path = os.path.join(
                    global_dir, f"global_{metric}.png"
                )
                plot_tasks.append((
                    metric,
                    global_counters[metric],
                    plot_path,
                    "Global",
                ))

        # Add global score histogram
        if global_score_counter:
            score_plot_path = os.path.join(global_dir, "global_score.png")
            # Use a pseudo-metric name for proper float bucket sorting
            plot_tasks.append((
                "overall_score",
                global_score_counter,
                score_plot_path,
                "Global",
            ))

        # Plot all histograms in parallel
        if plot_tasks:
            print(f"\nPlotting {len(plot_tasks)} histograms in parallel...")
            saved_plots = plot_histograms_parallel(plot_tasks, args.num_workers)
            print(f"Successfully saved {len(saved_plots)} histogram plots")


if __name__ == "__main__":
    main()
