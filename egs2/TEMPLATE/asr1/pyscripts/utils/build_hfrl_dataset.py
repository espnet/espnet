#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from espnet2.speechlm.definitions import SPEECHLM_TASKS

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Build the dataset from HFRL training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ref_json",
        type=Path,
        required=True,
        help="reference data.json file to specify task and token list",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="output data.json file",
    )
    parser.add_argument(
        "--path_modality_types",
        type=str,
        default=[],
        action="append",
        help="path_name_file of the new entry file",
    )
    parser.add_argument(
        "--metric_names",
        type=str,
        default=[],
        action="append",
        help="the metric names to ranking and select the samples",
    )
    parser.add_argument(
        "--metric_files",
        type=str,
        default=[],
        action="append",
        help="the metric files to ranking and select the samples",
    )
    parser.add_argument(
        "--metric_weights",
        type=float,
        default=[],
        action="append",
        help="the weigts of metrics to ranking and select the samples",
    )
    parser.add_argument(
        "--pos_threshold",
        type=float,
        default=0.2,
        help="ranking percentage lower than this threshold are positive exmaples",
    )
    parser.add_argument(
        "--neg_threshold",
        type=float,
        default=0.8,
        help="ranking percentage larger than this threshold are negative examples",
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    data_json = json.load(open(args.ref_json))
    task = data_json["task"]
    task_def = SPEECHLM_TASKS[task]

    # (1) load data files
    if len(task_def.targets) > 1:
        raise ValueError(f"Currently, we only support tasks with one target triplet.")

    all_triplets = task_def.data_triplets
    all_names = [e[0] for e in all_triplets]
    if len(all_triplets) != len(args.path_modality_types):
        raise ValueError(
            f"Task definition requires {len(all_triplets)} entries"
            f"But only {len(args.path_name_types)} specified"
        )

    data_dict = dict()
    for path_modality_type, name in zip(args.path_modality_types, all_names):
        logging.info(f"Entry {name} is specified as {path_modality_type}")

        path, modality, _type = path_modality_type.strip().split(",")
        if _type == "multicol_kaldi_ark":
            entry_dict = parse_multicol_kaldi_ark(path, task)
        else:
            entry_dict = parse_twocol_file(path)

        for utt_name, content in entry_dict.items():
            if utt_name not in data_dict:
                data_dict[utt_name] = dict()
            data_dict[utt_name][name] = content

    # (2) load all scoring info
    for path, metric in zip(args.metric_files, args.metric_names):
        logging.info(f"Add metric {metric} with file {path}")
        metric_dict = parse_json_file(path, task, metric)
        for utt_name, content in metric_dict.items():
            if utt_name in data_dict:
                data_dict[utt_name][metric] = content
            else:
                logging.info(
                    f"Get metric {metric} for {utt_name} but the example is missing"
                )

    # (3) clear incomplete examples
    incomplete_utts = []
    # (3.1) all entries should exist
    for utt_name, utt_dict in data_dict.items():
        for entry_name in all_names + args.metric_names:
            if entry_name not in utt_dict:
                incomplete_utts.append(utt_name)
                logging.info(f"utt {utt_name} is exclude as {entry_name} is missing")
                break
    for utt_name in incomplete_utts:
        del data_dict[utt_name]

    # (3.2) Some contents are dict for examples. Make their entries consistent
    for utt_name, utt_dict in data_dict.items():
        dict_entries = [
            key for key, value in utt_dict.items() if isinstance(value, dict)
        ]
        valid_examples = set(utt_dict[dict_entries[0]].keys())
        for entry in dict_entries:
            valid_examples = valid_examples & set(utt_dict[entry])
        for entry in dict_entries:
            utt_dict[entry] = {
                k: v for k, v in utt_dict[entry].items() if k in valid_examples
            }

    # (4) select the positive and negative examples
    pos_name = all_names[-1]
    neg_name = "sampled.scp"
    for utt_name, utt_dict in data_dict.items():
        pos_examples, neg_examples = sample_select(
            utt_dict,
            pos_threshold=args.pos_threshold,
            neg_threshold=args.neg_threshold,
            metrics=args.metric_names,
            metric_weights=args.metric_weights,
        )
        sample_dict = utt_dict[pos_name]
        pos_examples = " ".join(
            [v for k, v in sample_dict.items() if k in pos_examples]
        )
        neg_examples = " ".join(
            [v for k, v in sample_dict.items() if k in neg_examples]
        )
        utt_dict[pos_name] = pos_examples
        utt_dict[neg_name] = neg_examples

    # (5) write and dump
    data_files = []
    for path_modality_type, name in zip(args.path_modality_types, all_names):
        _, modality, _type = path_modality_type.strip().split(",")
        path = str(args.output_dir / name)
        data_files.append([path, modality, _type])

    # add additional file with same specification of the last entry
    # but the file name is "sampled.scp"
    data_files.append(data_files[-1].copy())
    data_files[-1][0] = data_files[-1][0].replace(all_names[-1], neg_name)
    all_names.append(neg_name)

    writers = {
        name: open(path, "w") for name, (path, _, _) in zip(all_names, data_files)
    }

    for utt_name, utt_dict in data_dict.items():
        for name, writer in writers.items():
            writer.write(f"{utt_name} {utt_dict[name]}\n")

    data_files = [",".join(e) for e in data_files]
    data_json["data_files"] = data_files
    data_json["num_examples"] = len(data_dict.keys())
    data_json["examples"] = list(data_dict.keys())

    with open(args.output_dir / "data.json", "wb") as writer:
        writer.write(
            json.dumps(data_json, indent=4, ensure_ascii=False, sort_keys=False).encode(
                "utf_8"
            )
        )
    logging.info(f"Save data.json file {args.output_dir}")


def sample_select(
    utt_dict,
    pos_threshold,
    neg_threshold,
    metrics,
    metric_weights,
):
    keys = set(utt_dict[metrics[0]].keys())
    scores = {key: 0.0 for key in keys}

    for metric, metric_weight in zip(metrics, metric_weights):
        metric_dict = utt_dict[metric]
        kvs = [(k, v) for k, v in metric_dict.items()]
        kvs.sort(key=lambda x: x[1], reverse=check_reverse(metric))

        prev_value = None
        count = 0
        for key, value in kvs:
            if value == prev_value:
                scores[key] += count * metric_weight
            else:
                prev_value = value
                count += 1
                scores[key] += count * metric_weight

    key_scores = [(k, s) for k, s in scores.items()]
    key_scores.sort(key=lambda x: x[1])

    n_sample = len(key_scores)
    pos_examples = [kv[0] for kv in key_scores[: int(pos_threshold * n_sample)]]
    neg_examples = [kv[0] for kv in key_scores[int(neg_threshold * n_sample) :]]

    return pos_examples, neg_examples


def check_reverse(metric):
    if metric in ["spk_similarity", "utmos"]:
        return True
    elif metric in ["edit_distance"]:
        return False
    else:
        raise NotImplementedError(f"Metric {metric} is not supported yet")


def parse_json_file(path, task, metric):
    ret_dict = dict()
    for line in open(path):
        line = line.strip().replace("'", '"')
        line = json.loads(line)
        example_name = line.pop("key")
        content = line.pop(metric)
        utt_name = example_name.lstrip(f"{task}_").split("_sample")[0]
        if utt_name not in ret_dict:
            ret_dict[utt_name] = dict()
        ret_dict[utt_name][example_name] = content

    return ret_dict


def parse_multicol_kaldi_ark(path, task):
    """will assume the data is organized in format <task>_<utt_name>_sampleN"""
    ret_dict = dict()
    for line in open(path):
        example_name, content = line.strip().split()
        utt_name = example_name.lstrip(f"{task}_").split("_sample")[0]
        if utt_name not in ret_dict:
            ret_dict[utt_name] = dict()
        ret_dict[utt_name][example_name] = content

    return ret_dict


def parse_twocol_file(path):
    ret_dict = dict()
    for line in open(path):
        utt_name, content = line.strip().split(maxsplit=1)
        ret_dict[utt_name] = content

    return ret_dict


if __name__ == "__main__":
    main()
