#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys


def get_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--src", type=str, help="", required=True)
    parser.add_argument("--dst", type=str, help="", required=True)
    parser.add_argument("--nj", type=int, help="", required=True)

    return parser


def main(args):
    args = get_parser().parse_args(args)

    lines = []
    idx = 0
    for j in range(args.nj):
        split_lines = open(
            args.src + "/out." + str(j + 1) + "/instances.log", "r"
        ).readlines()
        for i, line in enumerate(split_lines):
            local_idx = '"index": ' + str(i)
            global_idx = '"index": ' + str(idx)
            line = line.replace(local_idx, global_idx, 1)
            lines.append(line)
            idx += 1

    with open(args.dst, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main(sys.argv[1:])
