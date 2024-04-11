#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import os
import sys
import glob
import json

from espnet2.speechlm.definitions import tasks

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("build combined vocabulary")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Build the combined vocabulary for speechlm",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="output json file to specify the training data entires",
    )
    parser.add_argument(
        "--task", type=str, required=True, help="Task this dataset is prepared for"
    )
    parser.add_argument(
        "--file_modality_type",
        type=str,
        default=[],
        action="append",
        help="Append file triplet e.g. --file_modality_type <file>,<modality>,<type>",
    )
    parser.add_argument(
        "--token_list",
        type=str,
        default=[],
        action="append",
        help="Append token_list e.g. --token_list <token_list>",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    task_format = tasks[args.task]

    metadata = {}

    # (1) task
    metadata["task"] = args.task

    # (2) match all input files with the required files
    metadata["data_files"] = []
    all_entries_required = task_format.encoder_entries + task_format.decoder_entries
    all_entries_provided = [e.strip().split(",") for e in args.file_modality_type]
    for tgt_name, tgt_modality, tgt_type in all_entries_required:
        entry_found = False
        for name, modality, _type in all_entries_provided:
            if (
                modality == tgt_modality
                and _type == tgt_type
                and name.endswith(tgt_name)
            ):
                entry_found = True
                metadata["data_files"].append([name, modality, _type])
        if not entry_found:
            raise ValueError(f"No file with {tgt_name}, {tgt_modality}, {tgt_type}")

    # (3) record the vocabularies
    metadata["vocabularies"] = args.token_list

    # (4) ensure all examples are well-paired.
    example_dict = {}
    for file_triplet in metadata["data_files"]:
        file_path = file_triplet[0].strip().split(",")[0]
        feat_name = file_path.split("/")[-1]
        for line in open(file_path):
            example_id = line.strip().split()[0]
            if example_id not in example_dict:
                example_dict[example_id] = {}
            example_dict[example_id][feat_name] = None

    examples = []
    needed_names = [e[0] for e in all_entries_required]
    for example_id in example_dict.keys():
        for name in needed_names:
            if not name in example_dict[example_id]:
                raise ValueError(f"Example {example_id} doesn't have {name}")
        examples.append(example_id)

    metadata["num_examples"] = len(example_dict)
    metadata["examples"] = examples

    # (5) dump json
    metadata["data_files"] = [",".join(e) for e in metadata["data_files"]]
    with open(args.output_json, "wb") as writer:
        writer.write(
            json.dumps(metadata, indent=4, ensure_ascii=False, sort_keys=False).encode(
                "utf_8"
            )
        )


if __name__ == "__main__":
    main()
