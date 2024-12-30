#!/usr/bin/env python3

# Copyright 2024 Yuxun Tang
# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import json
import logging
import os
import sys
from pathlib import Path
import numpy as np

from espnet2.fileio.npy_scp import NpyScpWriter

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
        "--output_dir",
        type=Path,
        required=True,
        help="output directory",
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

    samples_dict = dict()
    # sample_dict = {
        # utt_name: {
            # sample_name: {
                # modality: value
    # }}}
    data_dict = dict() 
    # data_dict = {
        # utt_name: {
            # metric: {
                # sample_name: value
    # }}}
    samples_keys = []
    # (1) get all filenames
    for path_modality_type in args.path_modality_types:
        path, modality, _type = path_modality_type.strip().split(",")
        # NOTE(Yuxun): now we only have the key 'idx'.
        samples_keys.append(modality)

        if _type == "npy":
            entry_dict = parse_twocol_file(path, _type)
        else:
            raise NotImplementedError(f"Not support type {_type}.")

        for utt_name, content in entry_dict.items():
            if utt_name not in data_dict:
                data_dict[utt_name] = dict()
            if utt_name not in samples_dict:
                samples_dict[utt_name] = dict()
            for sample_name, value in content.items():
                if sample_name not in samples_dict[utt_name]:
                    samples_dict[utt_name][sample_name] = dict()
                samples_dict[utt_name][sample_name].update({modality: value})

    # (2) load scoring info
    for path, metric in zip(args.metric_files, args.metric_names):
        logging.info(f"Add metric {metric} with file {path}")
        metric_dict = parse_json_file(path, metric)
        for utt_name, content in metric_dict.items():
            if utt_name in data_dict:
                if metric not in data_dict[utt_name]:
                    data_dict[utt_name][metric] = dict()
                data_dict[utt_name][metric].update(content)
            else:
                logging.info(
                    f"Get metric {metric} for {utt_name} but the example is missing"
                )

    # (3) clear incomplete examples
    incomplete_utts = []
    # (3.1) all entries should exist
    for utt_name, utt_dict in data_dict.items():
        for entry_name in args.metric_names:
            if entry_name not in utt_dict:
                incomplete_utts.append(utt_name)
                logging.info(f"utt {utt_name} is exclude as {entry_name} is missing")
                break
    for utt_name in incomplete_utts:
        del data_dict[utt_name]

    # # (3.2) Some contents are dict for examples. Make their entries consistent
    # for utt_name, utt_dict in data_dict.items():
    #     dict_entries = [
    #         key for key, value in utt_dict.items() if isinstance(value, dict)
    #     ]
    #     valid_examples = set(utt_dict[dict_entries[0]].keys())
    #     for entry in dict_entries:
    #         valid_examples = valid_examples & set(utt_dict[entry])
    #     for entry in dict_entries:
    #         utt_dict[entry] = {
    #             k: v for k, v in utt_dict[entry].items() if k in valid_examples
    #         }

    # (4) select the positive and negative examples
    pos_name = "pos"
    neg_name = "neg"
    chosen_dict = {}
    pos_shape = {}
    neg_shape = {}
    for utt_name, utt_dict in data_dict.items():
        pos_examples, neg_examples = sample_select(
            utt_dict,
            pos_threshold=args.pos_threshold,
            neg_threshold=args.neg_threshold,
            metrics=args.metric_names,
            metric_weights=args.metric_weights,
        )
        sample_dict = samples_dict[utt_name]
        # print(f"pos: {pos_examples}")
        # print(f"neg: {neg_examples}")
        # print(sample_dict.keys())
        examples = [v for k, v in sample_dict.items() if k in pos_examples]
        pos_examples = {}
        pos_shape[utt_name] = dict()
        for key in samples_keys:
            feat_list = []
            for example in examples:
                feat_list.append(example[key])
            feat = np.stack(feat_list, axis=1)
            pos_examples[key] = feat # [T, K]
            pos_shape[utt_name][key] = [feat.shape[0]]
        examples = [v for k, v in sample_dict.items() if k in neg_examples]
        neg_examples = {}
        neg_shape[utt_name] = dict()
        for key in samples_keys:
            feat_list = []
            for example in examples:
                feat_list.append(example[key])
            feat = np.stack(feat_list, axis=1)
            neg_examples[key] = feat # [T, K]
            neg_shape[utt_name][key] = [feat.shape[0]]

        chosen_dict[utt_name] = dict()
        chosen_dict[utt_name].update({pos_name: pos_examples})
        chosen_dict[utt_name].update({neg_name: neg_examples})

    # (5) write and dump
    pos_npy_dir = os.path.join(args.output_dir, "eval_metrics", pos_name)
    if not os.path.exists(pos_npy_dir):
        os.makedirs(os.path.join(args.output_dir, "eval_metrics", pos_name))
    pos_writers = [
        NpyScpWriter(pos_npy_dir, os.path.join(args.output_dir, f"pos_samples_{name}")) for name in samples_keys
    ]
    pos_shape_writers = [
        open(os.path.join(args.output_dir, f"pos_{name}_shape"), "w") for name in samples_keys
    ]
    neg_npy_dir = os.path.join(args.output_dir, "eval_metrics", neg_name)
    if not os.path.exists(neg_npy_dir):
        os.makedirs(os.path.join(args.output_dir, "eval_metrics", neg_name))
    neg_writers = [
        NpyScpWriter(neg_npy_dir, os.path.join(args.output_dir, f"neg_samples_{name}")) for name in samples_keys
    ]
    neg_shape_writers = [
        open(os.path.join(args.output_dir, f"neg_{name}_shape"), "w") for name in samples_keys
    ]
    for key, pos_writer, neg_writer, pos_shape_writer, neg_shape_writer in zip(samples_keys, pos_writers, neg_writers, pos_shape_writers, neg_shape_writers):
        for utt_name, content in chosen_dict.items():
            pos_samples = content[pos_name][key]
            neg_samples = content[neg_name][key]
            pos_writer[utt_name] = pos_samples
            neg_writer[utt_name] = neg_samples
            pos_shape_writer.write("{} {}\n".format(utt_name, ",".join(map(str, pos_shape[utt_name][key]))))
            neg_shape_writer.write("{} {}\n".format(utt_name, ",".join(map(str, neg_shape[utt_name][key]))))
    logging.info(f"Save selected samples file {args.output_dir}")


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
    if metric in ["spk_similarity", "singmos"]:
        return True
    elif metric in ["mcd", "f0_rmse"]:
        return False
    else:
        raise NotImplementedError(f"Metric {metric} is not supported yet")


def parse_json_file(path, metric):
    """will assume the data is organized in format <utt_name>_N"""
    ret_dict = dict()
    for line in open(path):
        line = line.strip().replace("'", '"')
        line = json.loads(line)
        sample_name = line.pop("key")
        utt_name = "_".join(sample_name.split("_")[:-1])
        content = line.pop(metric)
        if utt_name not in ret_dict:
            ret_dict[utt_name] = dict()
        sample_dict = {sample_name: content}
        ret_dict[utt_name].update(sample_dict)

    return ret_dict


def parse_twocol_file(path, file_type):
    """will assume the data is organized in format <utt_name>"""
    ret_dict = dict()
    for line in open(path):
        sample_name, content = line.strip().split(maxsplit=1)
        if file_type == "npy":
            content = np.load(content)
        utt_name = "_".join(sample_name.split("_")[:-1])
        if utt_name not in ret_dict:
            ret_dict[utt_name] = dict()
        ret_dict[utt_name].update({sample_name: content})

    return ret_dict


if __name__ == "__main__":
    main()
