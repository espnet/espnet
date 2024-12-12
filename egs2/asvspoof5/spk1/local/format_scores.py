# format_scores.py
# This script formats the scores from the scoring script into
# the ASVspoof5 E2E SASV format
# the input file should be in the format:
# <enroll_id>*<test_id> <score>
# the output file will be a TSV file in the format:
# <spk-id> <testfile_id> <cm-score> <asv-score> <sasv-score>
# here cm and asv scores will not be used

import argparse
import os
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="Input file with scores")
    parser.add_argument("--output_file", help="Output file with formatted scores")
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    with open(input_file, "r") as f:
        lines = f.readlines()

    with open(output_file, "w") as f:
        # write header
        f.write("spk\tfilename\tcm-score\tasv-score\tsasv-score\n")
        for line in lines:
            trial_id, score = line.strip().split()
            spk_enroll_id, test_id = trial_id.split("*")
            f.write(f"{spk_enroll_id}\t{test_id}\t-\t-\t{score}\n")


if __name__ == "__main__":
    main()
