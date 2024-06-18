#!/usr/bin/env python3

import argparse
import editdistance
import json

from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse --metric and --score_dir options")
    parser.add_argument('--score_dir', type=Path, required=True, help='The directory for scores')

    args = parser.parse_args()
    return args

def main(args):
    ref_reader = open(args.score_dir / "ref.trn")
    hyp_reader = open(args.score_dir / "hyp.trn")
    writer = open(args.score_dir / "utt_result.txt", 'w')

    for ref_line, hyp_line in zip(ref_reader, hyp_reader):
        ref_name, ref_content = parse_trn(ref_line)
        hyp_name, hyp_content = parse_trn(hyp_line)
        ref_content = ref_content.split()
        hyp_content = hyp_content.split()

        if ref_name != hyp_name:
            raise ValueError(f"cannot compare {ref_line} vs. {hyp_line}")
        
        stat_dict = {
            "key": ref_name,
            "edit_distance": editdistance.eval(ref_content, hyp_content),
            "word_count": len(ref_content),
        }
        json.dump(stat_dict, writer)
        writer.write("\n")


def parse_trn(line):
    elems = line.strip().split()
    content = " ".join(elems[:-1])
    example_name = elems[-1].lstrip("(").rstrip(")")  

    return example_name, content

if __name__ == "__main__":
    args = parse_arguments()
    main(args)