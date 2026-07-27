#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Merge quality judge summaries from all detail levels."""

import argparse
import json
import os
from typing import Any, Dict


def merge_summaries(output_dir: str) -> Dict[str, Any]:
    """Merge summary files from all detail levels.

    Args:
        output_dir: Directory containing summary_{level}.json files.

    Returns:
        Combined summary dict with overall statistics.
    """
    combined: Dict[str, Any] = {"by_level": {}}

    for level in ["realistic", "imaginary"]:
        summary_file = os.path.join(output_dir, f"summary_{level}.json")
        if os.path.exists(summary_file):
            with open(summary_file, "r", encoding="utf-8") as f:
                combined["by_level"][level] = json.load(f)

    # Compute overall stats
    total = sum(
        d.get("total_samples", 0) for d in combined["by_level"].values()
    )
    passed = sum(
        d.get("passed_samples", 0) for d in combined["by_level"].values()
    )
    combined["total_samples"] = total
    combined["passed_samples"] = passed
    combined["overall_pass_rate"] = round(passed / total, 4) if total > 0 else 0

    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Merge quality judge summaries from all detail levels."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory containing summary files.",
    )
    args = parser.parse_args()

    combined = merge_summaries(args.output_dir)

    # Save combined summary
    output_file = os.path.join(args.output_dir, "summary_all.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    # Print results
    total = combined["total_samples"]
    passed = combined["passed_samples"]
    pass_rate = combined["overall_pass_rate"]
    print(f"Overall: {passed}/{total} passed ({pass_rate:.1%})")
    print(f"Summary saved to: {output_file}")


if __name__ == "__main__":
    main()
