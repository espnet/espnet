#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from typing import List, Set

import numpy as np
import scipy


def get_parser():
    parser = argparse.ArgumentParser(description="Universal evaluation script")
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
        type=bool,
        default=False,
        help="skip missing utterances",
    )
    return parser


def calculate_metrics(ref_metric_scores, pred_metric_scores, prefix="utt"):
    """Calculate utterance-level metrics."""
    if len(ref_metric_scores) != len(pred_metric_scores):
        raise ValueError(
            f"Number of utterances mismatch: {len(ref_metric_scores)} != {len(pred_metric_scores)}"
        )
    ref_metric_scores = np.array(ref_metric_scores)
    pred_metric_scores = np.array(pred_metric_scores)
    mse = np.mean((ref_metric_scores - pred_metric_scores) ** 2)
    lcc = np.corrcoef(ref_metric_scores, pred_metric_scores)[0, 1]
    srcc = scipy.stats.spearmanr(ref_metric_scores, pred_metric_scores)[0]
    ktau = scipy.stats.kendalltau(ref_metric_scores, pred_metric_scores)[0]
    return {
        "{}_mse".format(prefix): mse,
        "{}_lcc".format(prefix): lcc,
        "{}_srcc".format(prefix): srcc,
        "{}_ktau".format(prefix): ktau,
    }


def load_sys_info(sys_info_file: str):
    utt2sys = {}
    with open(sys_info_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            utt2sys[parts[0]] = parts[1]
    return utt2sys


def load_metrics(metrics_file, detect_metric_names=False):
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
            utt2metrics[utt] = json.loads(metrics)
            if detect_metric_names:
                metric_names = set(utt2metrics[utt].keys())
                metric_names.update(metric_names)

    return utt2metrics, metric_names


if __name__ == "__main__":
    args = get_parser().parse_args()
    ref_metrics, ref_metric_names = load_metrics(
        args.ref_metrics, detect_metric_names=True
    )
    pred_metrics, metric_names = load_metrics(
        args.pred_metrics, detect_metric_names=True
    )
    sys_info = load_sys_info(args.sys_info) if args.sys_info else None
    assert (
        sys_info is not None or args.level == "utt"
    ), "System information is required for system-level evaluation"
    final_result = {}
    for metric in metric_names:
        if metric not in ref_metric_names:
            logging.warning(f"Missing metric: {metric} in reference metric.scp")
        if args.level == "utt":
            pred_metric, ref_metric = [], []
        else:
            pred_metric, ref_metric = {}, {}
        for utt in pred_metrics.keys():
            # Checks for missing utterances and metrics
            if utt not in ref_metrics.keys():
                logging.warning(f"Missing utterance: {utt} in reference metric.scp")
            if metric not in pred_metrics[utt]:
                if args.skip_missing:
                    logging.warning(
                        f"Missing metric: {metric} in prediction metric.scp"
                    )
                    continue
                raise ValueError(f"Missing metric: {metric} in prediction metric.scp")
            if metric not in ref_metrics[utt]:
                if args.skip_missing:
                    logging.warning(f"Missing metric: {metric} in reference metric.scp")
                    continue
                raise ValueError(f"Missing metric: {metric} in reference metric.scp")
            if args.level == "utt":
                pred_metric.append(pred_metrics[utt][metric])
                ref_metric.append(ref_metrics[utt][metric])
            else:
                sys_id = sys_info[utt]
                if sys_id not in pred_metric:
                    pred_metric[sys_id] = []
                    ref_metric[sys_id] = []
                pred_metric[sys_id].append(pred_metrics[utt][metric])
                ref_metric[sys_id].append(ref_metrics[utt][metric])

        if args.level == "utt":
            eval_results = calculate_metrics(
                ref_metric, pred_metric, prefix="utt_{}".format(metric)
            )
        else:
            pred_sys_avg = []
            ref_sys_avg = []
            for sys_id in pred_metric.keys():
                sys_pred_metrics = np.array(pred_metric[sys_id])
                sys_ref_metrics = np.array(ref_metric[sys_id])
                sys_pred_avg = np.mean(sys_pred_metrics)
                sys_ref_avg = np.mean(sys_ref_metrics)
                pred_sys_avg.append(sys_pred_avg)
                ref_sys_avg.append(sys_ref_avg)
            eval_results = calculate_metrics(
                ref_sys_avg, pred_sys_avg, prefix="sys_{}".format(metric)
            )
        final_result.update(eval_results)
    with open(args.out_file, "w") as f:
        json.dump(final_result, f, indent=4)
    logging.info(f"Results saved to {args.out_file}")

# Example usage:
# python universa_eval.py --level utt --ref_metrics ref_metrics.scp --pred_metrics pred_metrics.scp --out_file result.json
