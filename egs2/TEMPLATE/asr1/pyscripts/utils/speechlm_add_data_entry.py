#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Add a new data entry to data json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_json",
        type=Path,
        required=True,
        help="output json file to specify the training data entires",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        required=True,
        help="output json file to specify the training data entires",
    )
    parser.add_argument(
        "--path_name_types",
        type=str,
        default=[],
        action="append",
        help="path_name_file of the new entry file",
    )
    parser.add_argument(
        "--extra_path_name_types",
        type=str,
        default=[],
        action="append",
        help="path_name_file of the new entry file",
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    json_dict = json.load(open(args.input_json))
    logging.info(f"adding entries to {str(args.input_json)} ...")

    all_data_files = []
    names = []
    for path_name_type in args.path_name_types + json_dict["data_files"]:
        path = path_name_type.strip().split(",")[0]
        name = path.strip().split("/")[-1]
        if name not in names:
            all_data_files.append(path_name_type)
            names.append(name)
        else:
            logging.info(f"Skipping {path_name_type} since it already exists")
    json_dict["data_files"] = all_data_files

    if len(args.extra_path_name_types) > 0:
        json_dict["extra_data_files"] = args.extra_path_name_types

    writer = open(args.output_json, "wb")
    writer.write(
        json.dumps(json_dict, indent=4, ensure_ascii=False, sort_keys=False).encode(
            "utf_8"
        )
    )
    logging.info(f"saving new data.json in {args.output_json}")


if __name__ == "__main__":
    main()
