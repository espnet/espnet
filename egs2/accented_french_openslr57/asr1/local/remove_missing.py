# This script removes very few examples (less than 0.1%) of the train and devtest sets
# Those few examples contained corrupted utterance Id or empty transcripts.

import argparse
import os

parser = argparse.ArgumentParser(description="Normalize test text.")
parser.add_argument("--folder", type=str, help="path of download folder")
parser.add_argument("--train", type=str, help="path of train folder")
parser.add_argument("--devtest", type=str, help="path of devtest folder")


def main(cmd=None):
    args = parser.parse_args(cmd)

    base = args.folder
    train = args.train
    devtest = args.devtest

    existing = []
    for _, _, f in os.walk(base + "speech/train/ca16"):
        for fi in f:
            existing.append(fi[:-4])

    old_f = open(base + train + "ca16_conv/transcripts.txt")
    new_f = open(
        base + train + "ca16_conv/new_transcripts.txt",
        "w",
    )

    for row in old_f:
        if row.split(" ")[0][:-4] in existing:
            new_f.write(row)

    old_f = open(base + train + "ca16_read/conditioned.txt")
    new_f = open(
        base + train + "ca16_read/new_conditioned.txt",
        "w",
    )

    for row in old_f:
        if row.split(" ")[0] in existing:
            new_f.write(row)

    existing = []
    for _, _, f in os.walk(base + "speech/devtest/ca16"):
        for fi in f:
            existing.append(fi[:-4])

    old_f = open(base + devtest + "ca16_read/conditioned.txt")
    new_f = open(
        base + devtest + "ca16_read/new_conditioned.txt",
        "w",
    )

    for row in old_f:
        if row.split(" ")[0] in existing:
            new_f.write(row)


if __name__ == "__main__":
    main()
