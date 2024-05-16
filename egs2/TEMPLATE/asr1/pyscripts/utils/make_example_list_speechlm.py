#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import json
import logging
import os
import sys

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("find all example list based on train_jsons")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Build the combined vocabulary for speechlm",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="output example_list file"
    )
    parser.add_argument(
        "--json_files",
        type=str,
        nargs="+",
        help="Append token_list e.g. --token_list <json1> <json2>",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    writer = open(args.output_file, "w")
    for json_file in args.json_files:
        read_handle = open(json_file, "r")
        json_dict = json.load(read_handle)
        examples = json_dict["examples"]
        task = json_dict["task"]
        for example in examples:
            writer.write(f"{task}_{example}\n")


if __name__ == "__main__":
    main()
