# make_eval_shape.py
# This script generates a speech_shape file for the eval set

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Generate speech_shape file for eval set"
    )
    parser.add_argument("--trial_label", type=str, help="Path to the trial label file")
    parser.add_argument(
        "--target_duration", type=float, help="Target duration for each utterance"
    )
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")
    args = parser.parse_args()

    trial_label = args.trial_label
    output_dir = args.output_dir

    trial_keys = []

    with open(trial_label, "r") as f:
        lines = f.readlines()
        # strip the newline character, and trailing and leading whitespaces
        lines = [line.strip() for line in lines]
        for line in lines:
            trial_keys.append(line.split()[0])

    with open(os.path.join(output_dir, "speech_shape"), "w") as f_out:
        for key in trial_keys:
            f_out.write(key + " " + str(int(args.target_duration * 16000.0)) + "\n")
    # copy the speech_shape to have speech_shape2, speech_shape3, speech_shape4
    for i in range(2, 5):
        with open(os.path.join(output_dir, "speech_shape" + str(i)), "w") as f_out:
            for key in trial_keys:
                f_out.write(key + " " + str(int(args.target_duration * 16000.0)) + "\n")


if __name__ == "__main__":
    main()
