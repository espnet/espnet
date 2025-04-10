#!/usr/bin/env python3

# Copyright 2023 Yuning Wu

"""Evaluate token prediction accuracy between generated and groundtruth token."""

import argparse
import logging
import os

import numpy as np

from espnet2.fileio.npy_scp import NpyScpReader
from espnet2.fileio.read_text import load_num_sequence_text


def get_parser() -> argparse.Namespace:
    """Get argument parser."""
    parser = argparse.ArgumentParser(description="Evaluate token prediction accuracy.")
    parser.add_argument(
        "gen_token_scp", type=str, help="Path of the scp file for generated tokens."
    )
    parser.add_argument(
        "gt_token_file",
        type=str,
        help="Path of the groundtruth token file.",
    )
    parser.add_argument("token_name", type=str, help="Name of discrete tokens.")
    parser.add_argument(
        "--outdir",
        type=str,
        help="Path of directory to write the results.",
    )
    parser.add_argument(
        "--discrete_token_layers",
        type=int,
        default=1,
        help="layers of discrete tokens",
    )
    parser.add_argument(
        "--mix_type",
        type=str,
        default="frame",
        help="multi token mix type, 'sequence' or 'frame'.",
    )
    return parser


def main():
    """Run token accuracy calculation."""
    args = get_parser().parse_args()

    acc_dict = {}
    len_all = 0
    names = args.token_name.split(" ")
    for name in names:
        acc_dict[name] = {}
    if args.discrete_token_layers > 1:
        acc_dict["multi"] = {}

    gt_token_reader = load_num_sequence_text(args.gt_token_file, loader_type="text_int")
    gen_token_reader = NpyScpReader(args.gen_token_scp)
    for key in gen_token_reader:
        gen_token = gen_token_reader[key].squeeze().tolist()
        gt_token = gt_token_reader[key]
        acc = sum(gen == gt for gen, gt in zip(gen_token, gt_token))
        acc = acc / len(gen_token)
        if args.discrete_token_layers == 1:
            acc_dict[args.token_name][key] = acc
            len_all += len(gen_token)
        else:
            acc_dict["multi"][key] = acc
            token_len = len(gen_token) / args.discrete_token_layers
            for i in range(len(names)):
                name = names[i]
                if args.mix_type == "frame":
                    name_gen_token = gen_token[i :: args.discrete_token_layers]
                    name_gt_token = gt_token[i :: args.discrete_token_layers]
                elif args.mix_type == "sequence":
                    name_gen_token = gen_token[i * token_len : (i + 1) * token_len]
                    name_gt_token = gt_token[i * token_len : (i + 1) * token_len]
                acc = sum(gen == gt for gen, gt in zip(name_gen_token, name_gt_token))
                acc = acc / token_len
                acc_dict[name][key] = acc

    for name in acc_dict:
        name_dir = os.path.join(args.outdir, name)
        os.makedirs(name_dir, exist_ok=True)
        with open(f"{name_dir}/utt2accuracy", "w") as f:
            for utt_id in sorted(acc_dict[name].keys()):
                acc = acc_dict[name][utt_id]
                f.write(f"{utt_id} {acc:.4f}\n")

        # calculate statistics
        mean_acc = np.mean(np.array([v for v in acc_dict[name].values()]))
        std_acc = np.std(np.array([v for v in acc_dict[name].values()]))
        logging.info(f"Average: {mean_acc:.4f} ± {std_acc:.4f}")

        with open(f"{name_dir}/accuracy_avg_result.txt", "w") as f:
            f.write(f"#utterances: {len(acc_dict[name])}\n")
            f.write(f"Average: {mean_acc:.4f} ± {std_acc:.4f}")


if __name__ == "__main__":
    main()
