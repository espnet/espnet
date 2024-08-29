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
        description="Build the combined vocabulary for speechlm",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True, help="output directory"
    )
    parser.add_argument(
        "--json_files",
        type=str,
        nargs="+",
        help="Append token_list e.g. --token_list <json1> <json2>",
    )
    parser.add_argument("--nj", type=int, default=1, help="number of splits")
    return parser


def split_one_data_json(json_file, nj, output_dir):
    # (1) load json file
    json_dict = json.load(open(json_file))
    data_files = json_dict["data_files"].copy()
    task = json_dict["task"]

    # (2) load all data files
    all_file_dict = {}
    for file_triplet in data_files:
        path, name, _type = file_triplet.split(",")
        file_dict = {}
        for line in open(path):
            utt, content = line.strip().split(maxsplit=1)
            file_dict[utt] = content
        all_file_dict[(path, name, _type)] = file_dict

    # (3) assign the utterances to each nj.
    # By default, we split all examples evenly.
    # When some files presents, we will keep some grouping properties.
    # E.g., when utt2spk exists, utterances from the same speaker will
    # be assigned to the same nj.
    splits = None
    for file_triplet in data_files:
        path, name, _type = file_triplet.split(",")
        if Path(path).name == "utt2spk":
            logging.info(f"Split by utt2spk, using file {path}")
            splits = split_by_utt2spk(
                all_file_dict[(path, name, _type)],
                nj,
            )

        if splits is not None:
            break

    if splits is None:
        splits = split_by_default(
            all_file_dict[(path, name, _type)],
            nj,
        )

    # (4) write to the disk
    (output_dir / f"split{nj}").mkdir(parents=True, exist_ok=True)
    for j in range(1, nj + 1):
        sub_dir = output_dir / f"split{nj}" / str(j)
        sub_dir.mkdir(parents=True, exist_ok=True)

        this_split = splits[j - 1]

        # write data files
        data_files = []
        for (path, name, _type), data_dict in all_file_dict.items():
            file_name = Path(path).name
            new_file_name = str(sub_dir / file_name)
            data_files.append(f"{new_file_name},{name},{_type}")
            writer = open(new_file_name, "w")
            for utt in this_split:
                writer.write(f"{utt} {data_dict[utt]}\n")
            writer.close()

        # write json files
        this_json = json_dict.copy()
        this_json["data_files"] = data_files
        this_json["examples"] = this_split
        this_json["num_examples"] = len(this_split)

        writer = open(sub_dir / f"data.json", "wb")
        writer.write(
            json.dumps(this_json, indent=4, ensure_ascii=False, sort_keys=False).encode(
                "utf_8"
            )
        )

    task = json_dict["task"]
    # To merge multiple dataset, add task as the prefix
    splits = [[task + "_" + utt for utt in split] for split in splits]

    return splits


def split_by_default(data_dict, nj):
    retval = [[] for _ in range(nj)]
    for idx, key in enumerate(data_dict.keys()):
        retval[idx % nj].append(key)
    return retval


def split_by_utt2spk(data_dict, nj):
    spk2utt = {}
    for utt, spk in data_dict.items():
        if spk not in spk2utt:
            spk2utt[spk] = []
        spk2utt[spk].append(utt)

    # always find the job with minimum number of examples
    retval = [[] for _ in range(nj)]
    for utt_list in spk2utt.values():
        lengths = [len(lst) for lst in retval]
        argmin = lengths.index(min(lengths))
        retval[argmin].extend(utt_list)
    return retval


def main():
    parser = get_parser()
    args = parser.parse_args()

    example_list = [[] for _ in range(args.nj)]
    for json_file in args.json_files:
        json_file = Path(json_file)
        example_splits = split_one_data_json(
            json_file,
            args.nj,
            args.output_dir,
        )
        for idx, split in enumerate(example_splits):
            example_list[idx].extend(split)

    for j in range(1, args.nj + 1):
        writer = open(args.output_dir / f"split{args.nj}" / str(j) /f"example_list", "w")
        for utt in example_list[j - 1]:
            writer.write(f"{utt}\n")


if __name__ == "__main__":
    main()
