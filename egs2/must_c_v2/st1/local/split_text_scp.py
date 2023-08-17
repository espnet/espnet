#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys


def get_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--text", type=str, help="", required=True)
    parser.add_argument("--scp", type=str, help="", required=True)
    parser.add_argument("--dst", type=str, help="", required=True)
    parser.add_argument("--nj", type=int, help="", required=True)

    return parser


def main(args):
    args = get_parser().parse_args(args)

    text_lines = open(args.text, "r").readlines()
    scp_lines = open(args.scp, "r").readlines()
    assert len(text_lines) == len(scp_lines)

    line_idx = 0
    lines_per = len(text_lines) // args.nj
    text_splits = [[] for _ in range(args.nj)]
    scp_splits = [[] for _ in range(args.nj)]
    for text, scp in zip(text_lines, scp_lines):
        split_idx = line_idx // lines_per
        if split_idx >= args.nj:
            split_idx = args.nj - 1
        text_splits[split_idx].append(text)
        scp_splits[split_idx].append(scp)
        line_idx += 1

    for j in range(args.nj):
        with open(args.dst + "/target." + str(j + 1), "w") as f:  # base 1
            f.writelines(text_splits[j])
        with open(args.dst + "/source." + str(j + 1), "w") as f:
            f.writelines(scp_splits[j])


if __name__ == "__main__":
    main(sys.argv[1:])
