#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# For SpeechLM style data, concat short examples into long examples

import argparse
import json
import logging
import os
import sys
import numpy as np
from pathlib import Path

from espnet2.fileio.read_text import read_2columns_text

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)

def get_parser():
    parser = argparse.ArgumentParser(
        description="Concat examples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--input_data_json",
        type=Path,
        required=True,
        help="Input data.json file",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="The output directory",
    )

    # hyper-parameters
    parser.add_argument(
        "--concat_method",
        type=str,
        default="number",
        choices=["number", "bucket"],
        help="The concat method",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default="4000",
        help="The length limit of each long-form example",
    )
    parser.add_argument(
        "--n_concat",
        type=int,
        default="2",
        help="The number of examples in concat, when concat_method == number",
    )

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # (1) load all files
    json_dict = json.load(open(args.input_data_json))
    data_triplets = json_dict["data_files"]
    data_triplets = [t.split(",") for t in data_triplets]
    length_dict = {e: None for e in json_dict["examples"]}
    task = json_dict["task"]

    shape_file = args.input_data_json.parent / "stats" / "dec_seq_shape"
    if not shape_file.is_file():
        raise ValueError(f"The input data.json doesn't have shape file: {shape_file}")
    for line in open(shape_file):
        uid, length = line.strip().split()
        uid = uid.removeprefix(f"{task}_")
        if uid in length_dict:
            length_dict[uid] = int(length)
    for uid, length in length_dict.items():
        if length is None:
            raise ValueError(f"UID {uid} is not in the shape file")
    
    # (2) grouping
    group_file = None
    for file, modality, _ in data_triplets:
        if modality == "spk":
            assert group_file is None
            group_file = file
            break
    
    if group_file is None:
        groups = [list(length_dict.keys())]
    else:
        groups = dict()
        for line in open(group_file):
            uid, gid = line.strip().split()
            if gid not in groups:
                groups[gid] = []
            groups[gid].append(uid)
        groups = list(groups.values())
    
    # (3) concat: assign uid for each long-form example
    concat_examples = []
    for group in groups:
        if args.concat_method == "number":
            concat_examples.extend(
                concat_by_number(
                    group,
                    length_dict=length_dict,
                    n_concat=args.n_concat,
                    max_len=args.max_len,
                )
            )
        elif args.concat_method == "bucket":
            concat_examples.extend(
                concat_by_bucket(
                    group,
                    length_dict=length_dict,
                    max_len=args.max_len,
                )
            )
        else:
            raise NotImplementedError
    
    # (4) concat: exact concat
    concat_data_triplets = []
    for idx, (file, modality, _type) in enumerate(data_triplets):
            
        # (4.1) concat content
        reader = read_2columns_text(file)
        writer = open(args.output_dir / Path(file).name, 'w')
        for uid_list in concat_examples:
            concat_uid = "_".join(uid_list)

            if modality in ["ssl", "codec", "codec_ssl", "text_bpe", "g2p"] and _type in ["kaldi_ark", "text"]:
                concat_content = " ".join([reader[uid] for uid in uid_list])
            elif modality in ["spk"] and _type in "text":
                concat_content = [reader[uid] for uid in uid_list]
                assert all([v == concat_content[0] for v in concat_content])
                concat_content = concat_content[0]
            else:
                raise NotImplementedError(f"Modality {modality} and type {_type} is not supported yet")

            writer.write(f"{concat_uid} {concat_content}\n")
        
        # (4.2) save length
        if idx == 0:
            (args.output_dir / "stats").mkdir(parents=True, exist_ok=True)
            length_writer = open(args.output_dir / "stats" / "dec_seq_shape", 'w')
            for uid_list in concat_examples:
                concat_uid = "_".join(uid_list)
                concat_length = sum([length_dict[uid] for uid in uid_list])
                length_writer.write(f"{task}_{concat_uid} {concat_length}\n")
        
        # (4.2) save new triplets
        if _type == "kaldi_ark":
            _type = "multicol_kaldi_ark"
        concat_file = str(args.output_dir / Path(file).name)
        concat_data_triplets.append(",".join([concat_file, modality, _type]))
    
    # (5) save the data.json
    json_dict["data_files"] = concat_data_triplets
    json_dict["examples"] = ["_".join(uid_list) for uid_list in concat_examples]
    json_dict["num_examples"] = len(json_dict["examples"])

    with open(args.output_dir / "data.json", "wb") as writer:
        writer.write(
            json.dumps(json_dict, indent=4, ensure_ascii=False, sort_keys=False).encode(
                "utf_8"
            )
        )

    
def concat_by_number(group, length_dict, n_concat, max_len):
    ans = []
    count = 0
    while count < len(group):
        example = group[count: min(count + n_concat, len(group))]
        if sum([length_dict[uid] for uid in example]) < max_len:
            ans.append(example)
        else:
            logging.info(
                f"The summed length of these examples exceeds {max_len}: {example} "
                f"So skip concatenating them"
            )
            for uid in example:
                ans.append([uid])
        
        count += n_concat
    return ans
            
def concat_by_bucket(group, length_dict, max_len):
    raise NotImplementedError
        

if __name__ == "__main__":
    main()