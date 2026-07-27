#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Filter Stage 3 dialogues based on Stage 4 quality scores."""

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

# Expected categories and dimensions (same as sft_judge_quality.py)
EXPECTED_STRUCTURE = {
    "user_request_quality": [
        "naturalness", "clarity", "specificity", "feasibility", "language_quality"
    ],
    "alignment": [
        "intent_match", "content_coverage", "no_contradictions",
        "scope_match", "transcription_match"
    ],
    "thinking_trace_quality": [
        "reasoning_coherence", "step_completeness", "no_hallucination",
        "appropriate_depth", "trace_caption_consistency"
    ],
    "caption_quality": [
        "descriptiveness", "realism", "coherence", "completeness"
    ],
    "training_value": [
        "learning_signal", "non_trivial", "persona_fit"
    ],
}


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_judge_results(judge_file: str) -> Dict[str, Dict[str, Any]]:
    """Load judge results and build lookup by example_id."""
    results = load_jsonl(judge_file)
    return {r["example_id"]: r for r in results}


def get_all_scores(judge_result: Dict[str, Any]) -> List[Tuple[str, float]]:
    """Extract all dimension scores from judge result.

    Returns list of (dimension_key, score) tuples.
    """
    scores = judge_result.get("scores", {})
    all_scores = []

    for category, dimensions in EXPECTED_STRUCTURE.items():
        if category in scores and isinstance(scores[category], dict):
            for dim in dimensions:
                if dim in scores[category]:
                    val = scores[category][dim]
                    if isinstance(val, dict) and "score" in val:
                        key = f"{category}.{dim}"
                        all_scores.append((key, val["score"]))

    return all_scores


def passes_threshold(
    judge_result: Dict[str, Any],
    min_score: float,
    avg_score: float,
) -> Tuple[bool, str, List[str]]:
    """Check if judge result passes threshold.

    Returns:
        (passes, rejection_reason, failing_dimensions)
    """
    all_scores = get_all_scores(judge_result)

    if not all_scores:
        return False, "no_scores", []

    scores_only = [s for _, s in all_scores]
    actual_min = min(scores_only)
    actual_avg = sum(scores_only) / len(scores_only)

    # Find dimensions failing min threshold
    failing_dims = [dim for dim, score in all_scores if score < min_score]

    if actual_min < min_score:
        return False, "min_score", failing_dims
    if actual_avg < avg_score:
        return False, "avg_score", failing_dims

    return True, "", []


def compute_rejection_stats(
    rejected_results: List[Tuple[Dict[str, Any], str, List[str]]],
) -> Dict[str, Any]:
    """Compute rejection statistics.

    Args:
        rejected_results: List of (judge_result, rejection_reason, failing_dims)
    """
    by_min_score = 0
    by_avg_score = 0
    by_dimension: Dict[str, int] = {}

    for _, reason, failing_dims in rejected_results:
        if reason == "min_score":
            by_min_score += 1
        elif reason == "avg_score":
            by_avg_score += 1

        for dim in failing_dims:
            by_dimension[dim] = by_dimension.get(dim, 0) + 1

    # Sort by count (descending)
    by_dimension = dict(
        sorted(by_dimension.items(), key=lambda x: x[1], reverse=True)
    )

    return {
        "by_min_score": by_min_score,
        "by_avg_score": by_avg_score,
        "by_dimension": by_dimension,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Filter dialogues based on quality scores."
    )
    parser.add_argument(
        "--dialogue_file",
        type=str,
        required=True,
        help="Path to Stage 3 dialogues (input).",
    )
    parser.add_argument(
        "--judge_file",
        type=str,
        required=True,
        help="Path to Stage 4 judge results (input).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Stage 5 output directory.",
    )
    parser.add_argument(
        "--detail_level",
        type=str,
        required=True,
        choices=["realistic", "imaginary"],
        help="Detail level being filtered.",
    )
    parser.add_argument(
        "--min_score",
        type=float,
        default=3.0,
        help="Minimum score threshold (default: 3).",
    )
    parser.add_argument(
        "--avg_score",
        type=float,
        default=3.5,
        help="Average score threshold (default: 3.5).",
    )
    args = parser.parse_args()

    # Load data
    print(f"Loading dialogues from {args.dialogue_file}...")
    dialogues = load_jsonl(args.dialogue_file)
    print(f"Loaded {len(dialogues)} dialogues")

    print(f"Loading judge results from {args.judge_file}...")
    judge_lookup = load_judge_results(args.judge_file)
    print(f"Loaded {len(judge_lookup)} judge results")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Filter dialogues
    passed_dialogues = []
    rejected_results = []
    missing_judge = 0

    for dialogue in dialogues:
        example_id = dialogue.get("example_id")
        if example_id not in judge_lookup:
            missing_judge += 1
            continue

        judge_result = judge_lookup[example_id]
        passes, reason, failing_dims = passes_threshold(
            judge_result, args.min_score, args.avg_score
        )

        if passes:
            passed_dialogues.append(dialogue)
        else:
            rejected_results.append((judge_result, reason, failing_dims))

    # Write filtered dialogues
    output_file = os.path.join(
        args.output_dir, f"filtered_{args.detail_level}.jsonl"
    )
    with open(output_file, "w", encoding="utf-8") as f:
        for dialogue in passed_dialogues:
            f.write(json.dumps(dialogue, ensure_ascii=False) + "\n")

    print(f"Filtered dialogues saved to: {output_file}")

    # Compute and write summary
    rejection_breakdown = compute_rejection_stats(rejected_results)

    total_input = len(dialogues)
    total_passed = len(passed_dialogues)
    total_rejected = len(rejected_results)

    summary = {
        "total_input": total_input,
        "total_passed": total_passed,
        "total_rejected": total_rejected,
        "missing_judge_results": missing_judge,
        "pass_rate": round(total_passed / total_input, 4) if total_input > 0 else 0,
        "thresholds": {
            "min_score": args.min_score,
            "avg_score": args.avg_score,
        },
        "rejection_breakdown": rejection_breakdown,
    }

    summary_file = os.path.join(
        args.output_dir, f"summary_{args.detail_level}.json"
    )
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {summary_file}")
    print(f"\nResults for {args.detail_level}:")
    print(f"  Total input: {total_input}")
    print(f"  Passed: {total_passed}")
    print(f"  Rejected: {total_rejected}")
    if missing_judge > 0:
        print(f"  Missing judge results: {missing_judge}")
    print(f"  Pass rate: {summary['pass_rate']:.1%}")


if __name__ == "__main__":
    main()
