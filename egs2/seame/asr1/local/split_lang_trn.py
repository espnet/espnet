#!/usr/bin/env python3
# -*- encoding: utf8 -*-

import argparse
import os

from preprocess import (
    extract_mandarin_only,
    extract_non_mandarin,
    insert_space_between_mandarin,
    remove_redundant_whitespaces,
)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--trn", "-t", type=str, help=".trn file")
    parser.add_argument("--out", "-o", type=str, help="Output dir.")
    args = parser.parse_args()

    out_name = args.trn.split("/")[-1]  # hyp.trn / ref.trn
    eng_out_path = os.path.join(args.out, out_name + ".eng")
    man_out_path = os.path.join(args.out, out_name + ".man")

    with open(args.trn, "r") as fp:
        with open(eng_out_path, "w") as fp_eng:
            with open(man_out_path, "w") as fp_man:
                for line in fp:
                    sent, idx = line.split("\t")

                    sent_eng = extract_non_mandarin(sent)
                    sent_man = extract_mandarin_only(sent)
                    sent_man = insert_space_between_mandarin(sent_man)
                    sent_eng = remove_redundant_whitespaces(sent_eng)
                    sent_man = remove_redundant_whitespaces(sent_man)

                    fp_eng.write(sent_eng + "\t" + idx)
                    fp_man.write(sent_man + "\t" + idx)
