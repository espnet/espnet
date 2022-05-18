#!/usr/bin/env python

# Copyright 2022  Shanghai Jiao Tong University (Authors: Wangyou Zhang)
# Apache 2.0
import argparse
import random
from collections import defaultdict
from pathlib import Path

from split_train_dev import int_or_float_or_numstr, split_train_dev, split_train_dev_v2


def get_parser():
    """Argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("datalist", type=str, help="Path to the list of audio files")
    parser.add_argument(
        "--num_dev",
        type=int_or_float_or_numstr,
        required=True,
        help="Number of samples to assign to the development set "
        "(can be an integer, a float number, or a numeric string like '1/3')",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="{}.lst",
        help="A template path for storing output",
    )
    parser.add_argument(
        "--delim",
        type=str,
        default="_",
        help="Delimiter for determining the prefix for grouping",
    )
    parser.add_argument(
        "--prefix_num",
        type=int,
        default=3,
        help="Number of preceding splits (after delimiting) to form the prefix",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=True,
        choices=("same_size_group", "similar_size_group"),
        help="'same_size_group': sample from same-size groups "
        "to gather `num_dev` samples\n"
        "'similar_size_group': sample from similar-size groups "
        "to gather `num_dev` samples",
    )
    parser.add_argument(
        "--allowed_deviation",
        type=int,
        default=0,
        help="how many samples are allowed for the final dev split to be less than "
        "or more than the specified `num_dev`",
    )
    parser.add_argument(
        "--max_solutions",
        type=int,
        default=50,
        help="Maximum number of possible coin change solutions to search (only for "
        "mode='similar_size_group')",
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    random.seed(args.seed)

    datalist = Path(args.datalist).expanduser().resolve()
    all_data = defaultdict(list)
    with datalist.open("r") as f:
        for fpath in f:
            if not fpath:
                continue
            # e.g. 3.00_7.76_3.00_1.06_3.47_289.5578_240.7477_0.7049
            fname = Path(fpath).stem
            group_id = args.delim.join(fname.split(args.delim)[: args.prefix_num])
            all_data[group_id].append(fpath)

    if args.mode == "same_size_group":
        split_train_dev(
            all_data,
            args.num_dev,
            args.outfile,
        )
    elif args.mode == "similar_size_group":
        split_train_dev_v2(
            all_data,
            args.num_dev,
            args.outfile,
            allowed_deviation=args.allowed_deviation,
            max_solutions=args.max_solutions,
        )
    else:
        raise ValueError("Unsupported mode: %s" % args.mdoe)
