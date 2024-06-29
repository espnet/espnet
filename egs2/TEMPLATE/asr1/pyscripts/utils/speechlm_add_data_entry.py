#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import sys
import json
import argparse
import logging
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
        "--path",
        type=str,
        default=[],
        nargs="*",
        help="path of the new entry file",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=[],
        nargs="*",
        help="name of the new entry file",
    )
    parser.add_argument(
        "--type",
        type=str,
        default=[],
        nargs="*",
        help="type of the new entry file",
    )
    parser.add_argument(
        "--path_name_type",
        type=str,
        default=[],
        nargs="*",
        help="path_name_file of the new entry file",
    )

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    json_dict = json.load(open(args.input_json))
    logging.info(f"adding entries to {str(args.input_json)} ...")

    if "extra_data_files" not in json_dict:
        json_dict["extra_data_files"] = []

    for path, name, _type in zip(args.path, args.name, args.type):
        path_name_type = f"{path},{name},{_type}"
        json_dict["extra_data_files"].append(path_name_type)
        logging.info(f"add entry: {path_name_type}")
    
    for path_name_type in args.path_name_type:
        json_dict["extra_data_files"].append(path_name_type)
        logging.info(f"add entry: {path_name_type}")
    
    writer = open(args.output_json, 'wb')
    writer.write(
        json.dumps(
            json_dict, 
            indent=4, 
            ensure_ascii=False, 
            sort_keys=False)
            .encode("utf_8")
    )
    logging.info(f"saving new data.json in {args.output_json}")

if __name__ == "__main__":
    main()