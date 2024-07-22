#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import json
import logging
from pathlib import Path

from espnet2.speechlm.definitions import tasks


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

    # (1) collect metadata.
    # (1.1) task
    metadata["task"] = args.task

    # (1.2) vocabularies, if any
    if getattr(args, "token_list", None) is not None:
        metadata["vocabularies"] = args.token_list

    # (2) make sure all examples are ordered and paired
    # (2.1) match all input files with the required files
    file_triplets = []
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
                file_triplets.append([name, modality, _type])
        if not entry_found:
            raise ValueError(f"No triplet: {tgt_name},{tgt_modality},{tgt_type}")

    # (2.2) load all data entries
    example_dict = {}
    for file_triplet in file_triplets:
        file_path = file_triplet[0]
        feat_name = file_path.split("/")[-1]
        for line in open(file_path):
            example_id, content = line.strip().split(maxsplit=1)
            if example_id not in example_dict:
                example_dict[example_id] = {}
            example_dict[example_id][feat_name] = content

    # (2.3) find all examples that are well-paired.
    valid_example_ids = []
    needed_names = [e[0] for e in all_entries_required]
    for example_id in example_dict.keys():
        if all([name in example_dict[example_id] for name in needed_names]):
            valid_example_ids.append(example_id)
        else:
            logging.warning(f"Example {example_id} is not complete")

    logging.info(f"Keep {len(valid_example_ids)} out of {len(example_dict)} examples")

    # (3) dump each entry only for valid examples. All entries are ordered.
    entry_path = Path(args.output_json).parent / "entries"
    entry_path.mkdir(parents=True, exist_ok=True)
    writers = {name: open(entry_path / name, "w") for name in needed_names}
    for example_id in valid_example_ids:
        for name in needed_names:
            writers[name].write(f"{example_id} {example_dict[example_id][name]}\n")

    metadata["data_files"] = []
    for name, modality, _type in all_entries_required:
        file_path = str(entry_path / name)
        triplet = f"{file_path},{modality},{_type}"
        metadata["data_files"].append(triplet)

    metadata["num_examples"] = len(valid_example_ids)
    metadata["examples"] = valid_example_ids

    # dump json
    with open(args.output_json, "wb") as writer:
        writer.write(
            json.dumps(metadata, indent=4, ensure_ascii=False, sort_keys=False).encode(
                "utf_8"
            )
        )


if __name__ == "__main__":
    main()
