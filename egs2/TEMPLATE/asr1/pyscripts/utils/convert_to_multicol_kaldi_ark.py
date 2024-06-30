#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
from pathlib import Path

def get_parser():
    parser = argparse.ArgumentParser(
        description="Add a new data entry to data json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_scp",
        type=Path,
        required=True,
        help="input scp file",
    )
    parser.add_argument(
        "--output_scp",
        type=Path,
        required=True,
        help="output scp file",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
    )

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    data_dict = {}
    for line in open(args.input_scp):
        extend_name, content = line.strip().split(maxsplit=1)
        name = extend_name.lstrip(f"{args.task}_")
        name = name.split("_sample")[0]

        if name not in data_dict:
            data_dict[name] = []
        data_dict[name].append(content)
    
    writer = open(args.output_scp, 'w')
    for key, contents in data_dict.items():
        string = [key] + contents
        string = " ".join(string)
        writer.write(f"{string}\n")

if __name__ == "__main__":
    main()