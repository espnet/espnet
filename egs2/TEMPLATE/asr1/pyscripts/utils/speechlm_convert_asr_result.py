#!/usr/bin/env python3

import argparse
import editdistance
import json
import logging

from pathlib import Path
from espnet.utils.cli_utils import get_commandline_args

def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse --metric and --score_dir options")
    parser.add_argument('--ref_file', type=Path, required=True, help='reference file')
    parser.add_argument('--hyp_file', type=Path, required=True, help='hypothesis file')
    parser.add_argument('--out_file', type=Path, required=True, help='output file')
    parser.add_argument('--file_type', type=str, default=None, help='type of input file')

    args = parser.parse_args()
    return args

def main(args):
    ref_reader = open(args.ref_file)
    hyp_reader = open(args.hyp_file)
    writer = open(args.out_file, 'w')

    for ref_line, hyp_line in zip(ref_reader, hyp_reader):
        ref_name, ref_content = parse_line(ref_line, args.file_type)
        hyp_name, hyp_content = parse_line(hyp_line, args.file_type)
        ref_content = ref_content.split()
        hyp_content = hyp_content.split()

        if ref_name != hyp_name:
            raise ValueError(f"cannot compare {ref_line} vs. {hyp_line} | with utt: {ref_name} vs {hyp_name}")
        
        stat_dict = {
            "key": ref_name,
            "edit_distance": editdistance.eval(ref_content, hyp_content),
            "word_count": len(ref_content),
        }
        json.dump(stat_dict, writer)
        writer.write("\n")


def parse_line(line, file_type):
    
    line = line.strip().split()

    # by defualt, <example_name> <content> format
    if file_type is None:
        example_name = line[0]
        content = " ".join(line[1:])
    
    # trn: <content> (<example_name>)
    elif file_type == "trn":
        content = " ".join(line[:-1])
        example_name = line[-1].lstrip("(").rstrip(")")
    else:
        raise NotImplementedError

    return example_name, content

if __name__ == "__main__":
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)
    logging.info(get_commandline_args())
    args = parse_arguments()
    main(args)