#!/usr/bin/env python3

# Copyright 2021  Siddhant Arora
#           2021  Carnegie Mellon University
# Apache 2.0


import os
import sys
import logging
from collections import defaultdict

if __name__ == "__main__":
    # check numer of inputs
    if len(sys.argv) != 4:
        print("Usage: python multilingual_detokenizer.py [language-ID-file] [input-file] [output-file]")
        print("Example: python multilingual_detokenizer.py tgt_file.txt ref.trn ref.trn.detok")
        sys.exit(1)

    # read language-IDs and corresponding input lines
    with open(sys.argv[1]) as f:
        lang_ids = [line.rsplit()[1][1:-1] for line in f.readlines()]
    with open(sys.argv[2]) as f:
        input_lines = f.readlines()

    # group input lines to their corresponding language-IDs
    lang_lines = defaultdict(list)
    lang_indices = defaultdict(list)
    for ix, (lang_id, input_line) in enumerate(zip(lang_ids, input_lines)):
        lang_lines[lang_id].append(input_line)
        lang_indices[lang_id].append(ix)

    # perform detokenization for each language-ID
    input_dirname = os.path.dirname(sys.argv[2])
    input_basename = os.path.basename(sys.argv[2])
    output_dirname = os.path.dirname(sys.argv[3])
    output_basename = os.path.basename(sys.argv[3])
    output_lines = [""] * len(input_lines)

    for lang_id in lang_lines:
        lang_input_file = os.path.join(input_dirname, f"{lang_id}.{input_basename}")
        lang_output_file = os.path.join(output_dirname, f"{lang_id}.{output_basename}")
        with open(lang_input_file, "w") as f:
            f.writelines("".join(lang_lines[lang_id]))
        os.popen(f"detokenizer.perl -l {lang_id} -q < {lang_input_file} >> {lang_output_file}").read()

        with open(lang_output_file) as f:
            lang_output_lines = f.readlines()
            for ix, line in zip(lang_indices[lang_id], lang_output_lines):
                output_lines[ix] = line
        os.remove(lang_input_file)
        os.remove(lang_output_file)

    # write detokenized lines to output file
    with open(sys.argv[3], "w") as f:
        f.writelines("".join(output_lines))
