# generate_trial_file.py

# This script generates trials from the DeepMine sample subset for it's
#  use as a test/dev set

import argparse
import os
import sys
from itertools import combinations


def generate_trials(input_file, output_file):
    # Read data from the input file
    with open(input_file, "r") as f:
        file_paths = f.read().splitlines()

    # Generate speaker verification trials
    trials = []
    for path1, path2 in combinations(file_paths, 2):
        spk1 = path1.split("/")[0]
        spk2 = path2.split("/")[0]
        label = 1 if spk1 == spk2 else 0
        trials.append(f"{label} {path1}.wav {path2}.wav\n")

    # Write the trials to the output file
    with open(output_file, "w") as f:
        f.writelines(trials)

    print(f"SPK trials generated and written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SPK Trials")
    parser.add_argument(
        "--src", type=str, required=True, help="Input file with file paths"
    )
    parser.add_argument(
        "--dst",
        type=str,
        required=True,
        help="Output file to write speaker verification trials",
    )
    args = parser.parse_args()

    generate_trials(args.src, args.dst)
