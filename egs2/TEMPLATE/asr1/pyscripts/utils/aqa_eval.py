#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""Evaluate paired numeric metrics at utterance or system level."""

import argparse
import json
import logging

import numpy as np
import scipy.stats

from espnet2.utils.types import str2bool


def get_parser():
    """Build the numeric metric evaluation command-line parser."""
    parser = argparse.ArgumentParser(description="Evaluate numeric metric scores")
    parser.add_argument(
        "--level",
        type=str,
        default="utt",
        choices=["utt", "sys"],
    )
    parser.add_argument(
        "--ref_metrics",
        type=str,
        required=True,
        help="reference metrics file",
    )
    parser.add_argument(
        "--pred_metrics",
        type=str,
        required=True,
        help="metrics prediction file",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        required=True,
        help="output file",
    )
    parser.add_argument(
        "--sys_info",
        type=str,
        default=None,
        help="system information file",
    )
    parser.add_argument(
        "--skip_missing",
        type=str2bool,
        default=False,
        help="skip missing utterances",
    )
    return parser


def calculate_metrics(ref_metric_scores, pred_metric_scores, prefix="utt"):
    """Calculate utterance-level metrics."""
    if len(ref_metric_scores) != len(pred_metric_scores):
        raise ValueError(
            "Num of utt mismatch: "
            f"{len(ref_metric_scores)} != {len(pred_metric_scores)}"
        )
    if not len(ref_metric_scores):
        raise ValueError("No matching scores to evaluate")
    ref_metric_scores = np.asarray(ref_metric_scores, dtype=float)
    pred_metric_scores = np.asarray(pred_metric_scores, dtype=float)
    if (
        ref_metric_scores.ndim != 1
        or pred_metric_scores.ndim != 1
        or not np.isfinite(ref_metric_scores).all()
        or not np.isfinite(pred_metric_scores).all()
    ):
        raise ValueError("Metric scores must be finite numeric scalars")
    mse = float(np.mean((ref_metric_scores - pred_metric_scores) ** 2))
    # Correlation is undefined for singleton or constant score vectors.
    lcc = srcc = ktau = None
    if (
        len(ref_metric_scores) > 1
        and np.ptp(ref_metric_scores) > 0
        and np.ptp(pred_metric_scores) > 0
    ):
        lcc = float(np.corrcoef(ref_metric_scores, pred_metric_scores)[0, 1])
        srcc = float(scipy.stats.spearmanr(ref_metric_scores, pred_metric_scores)[0])
        ktau = float(scipy.stats.kendalltau(ref_metric_scores, pred_metric_scores)[0])
    return {
        f"{prefix}_mse": mse,
        f"{prefix}_lcc": lcc,
        f"{prefix}_srcc": srcc,
        f"{prefix}_ktau": ktau,
    }


def load_sys_info(sys_info_file: str):
    """Load unique utterance-to-system assignments."""
    utt2sys = {}
    with open(sys_info_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2 or parts[0] in utt2sys:
                raise ValueError(f"Invalid or duplicate system mapping: {line}")
            utt2sys[parts[0]] = parts[1]
    return utt2sys


def load_metrics(metrics_file, detect_metric_names=False):
    """Load unique utterance-keyed JSON objects and optionally collect names."""
    utt2metrics = {}
    metric_names = set()
    with open(metrics_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Invalid line: {line}")
            utt, metrics = parts
            if utt in utt2metrics:
                raise ValueError(f"Duplicate utterance: {utt} in {metrics_file}")
            values = json.loads(metrics)
            if not isinstance(values, dict):
                raise ValueError(f"Expected a JSON object for {utt} in {metrics_file}")
            utt2metrics[utt] = values
            if detect_metric_names:
                metric_names.update(utt2metrics[utt])

    return utt2metrics, metric_names


def main():
    """Evaluate matched numeric scores at utterance or system level.

    Strict mode requires identical utterance and metric coverage. With
    --skip_missing, evaluate only paired scores and omit metrics with no pairs.
    Undefined correlations are written as JSON null; no matched scores is an error.
    """
    args = get_parser().parse_args()
    ref_metrics, ref_metric_names = load_metrics(args.ref_metrics, True)
    pred_metrics, pred_metric_names = load_metrics(args.pred_metrics, True)
    sys_info = load_sys_info(args.sys_info) if args.sys_info else None
    if args.level == "sys" and sys_info is None:
        raise ValueError("System information is required for system-level evaluation")
    utterances = sorted(ref_metrics.keys() | pred_metrics.keys())
    if not args.skip_missing:
        for utt in utterances:
            for source, scores in (
                ("reference", ref_metrics),
                ("prediction", pred_metrics),
            ):
                if utt not in scores:
                    raise ValueError(f"Missing utterance: {utt} in {source} metric.scp")
    final_result = {}
    for metric in sorted(ref_metric_names | pred_metric_names):
        ref_scores, pred_scores = [], []
        systems = {}
        for utt in utterances:
            for source, scores in (
                ("reference", ref_metrics),
                ("prediction", pred_metrics),
            ):
                if metric not in scores.get(utt, {}):
                    if args.skip_missing:
                        break
                    raise ValueError(
                        f"Missing metric: {metric} for {utt} in {source} metric.scp"
                    )
            else:
                ref_value = ref_metrics[utt][metric]
                pred_value = pred_metrics[utt][metric]
                for value in (ref_value, pred_value):
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not np.isfinite(value)
                    ):
                        raise ValueError(
                            f"Metric {metric} for {utt} must be a finite numeric scalar"
                        )
                if args.level == "sys":
                    if utt not in sys_info:
                        if args.skip_missing:
                            continue
                        raise ValueError(
                            f"Missing system information for utterance: {utt}"
                        )
                    system_ref, system_pred = systems.setdefault(
                        sys_info[utt], ([], [])
                    )
                    system_ref.append(ref_value)
                    system_pred.append(pred_value)
                else:
                    ref_scores.append(ref_value)
                    pred_scores.append(pred_value)
        if args.level == "sys":
            for system_ref, system_pred in systems.values():
                ref_scores.append(np.mean(system_ref))
                pred_scores.append(np.mean(system_pred))
        if not ref_scores:
            logging.warning("No matching scores for metric %s; skipping", metric)
            continue
        final_result.update(
            calculate_metrics(ref_scores, pred_scores, prefix=f"{args.level}_{metric}")
        )
    if not final_result:
        raise ValueError("No matching scores to evaluate")
    with open(args.out_file, "w") as f:
        json.dump(final_result, f, indent=4, allow_nan=False)
    logging.info("Results saved to %s", args.out_file)


if __name__ == "__main__":
    main()
