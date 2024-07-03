# format_scores.py
# This script formats the scores from the scoring script into
# the ASVspoof5 E2E SASV format
# the input file should be in the format:
# <enroll_id>*<test_id> <score>	
# the output file will be a TSV file in the format:
# <test_id> <cm-score> <asv-score> <sasv-score>
# here cm and asv scores will not be used

import sys
import os
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="Input file with scores")
    parser.add_argument("--output_file", help="Output file with formatted scores")
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    with open(input_file, 'r') as f:
        lines = f.readlines()

    with open(output_file, 'w') as f:
        # write header
        f.write("filename\tcm-score\tasv-score\tsasv-score\n")
        for line in lines:
            trial_id, score = line.strip().split()
            enroll_id, test_id = trial_id.split('*')
            f.write(f"{test_id}\t-\t-\t{score}\n")


if __name__ == "__main__":
    main()
