#!/usr/bin/env python

import argparse
import functools
import os
import subprocess

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Split train, dev, test")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dev_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num_test_speakers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--num_test_samples",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--num_dev_samples",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--overwrite_data_dirs",
        type=bool,
        default=False,
    )

    MALE_VOICES = [
        "s124",
        "s176",
        "s178",
        "s181",
        "s188",
        "s206",
        "s220",
        "s225",
        "s234",
        "s235",
        "s157",
        "s162",
        "s216",
        "s222",
        "s223",
        "s231",
        "s236",
        "s240",
        "s250",
        "s273",
    ]
    FEMALE_VOICES = [
        "s146",
        "s180",
        "s186",
        "s208",
        "s209",
        "s214",
        "s215",
        "s221",
        "s264",
        "s268",
        "s169",
        "s185",
        "s187",
        "s200",
        "s226",
        "s228",
        "s247",
        "s251",
        "s256",
        "s258",
    ]
    args = parser.parse_args()

    if any([os.path.exists(x) for x in [args.train_dir, args.dev_dir, args.test_dir]]):
        if not args.overwrite_data_dirs:
            print("Directories already present. Cowardly not overwriting.")
            print("Set --overwrite_data_dirs flag to True to overwrite")
            exit(1)

    all_utts = set()
    with open(os.path.join(args.data_dir, "spk2utt"), "r", encoding="utf8") as f:
        lines = f.readlines()
        spk2utt = {line.split()[0]: line.split()[1:] for line in lines}
        for utts in spk2utt.values():
            all_utts.update(utts)

    prompts = {}

    with open(os.path.join(args.data_dir, "text"), "r", encoding="utf8") as f:
        text = f.readlines()
        for line in text:
            rec_id, prompt = line.split(maxsplit=1)
            if prompt not in prompts.keys():
                prompts[prompt] = []
            prompts[prompt].append(rec_id)

    test_utts_spread = set()

    while len(test_utts_spread) < args.num_test_samples:
        prompt = np.random.choice(list(prompts.keys()))
        test_utts_spread.update(prompts[prompt])
        del prompts[prompt]

    male_test_speakers = (args.num_test_speakers // 2) + (args.num_test_speakers % 2)
    female_test_speakers = args.num_test_speakers // 2
    test_speakers = np.concatenate(
        (
            np.random.choice(MALE_VOICES, male_test_speakers),
            np.random.choice(FEMALE_VOICES, female_test_speakers),
        )
    )
    print(f"test_speakers: {test_speakers}")
    test_utts = functools.reduce(
        lambda a, b: np.concatenate((a, spk2utt[b])), test_speakers, np.ndarray(0)
    )

    # test_utts = set(np.concatenate(list(spk2utt.values())[:args.num_test_speakers]))
    test_utts = set(test_utts).union(test_utts_spread)
    dev_utts = set()

    while len(dev_utts) < args.num_dev_samples:
        prompt = np.random.choice(list(prompts.keys()))
        dev_utts.update(prompts[prompt])
        del prompts[prompt]

    for utt in dev_utts:
        if utt in test_utts:
            test_utts.remove(utt)

    # all_utts = set(np.concatenate(list(spk2utt.values())))

    train_utts = all_utts - test_utts - dev_utts

    if not os.path.exists(args.train_dir):
        os.mkdir(args.train_dir)
    if not os.path.exists(args.dev_dir):
        os.mkdir(args.dev_dir)
    if not os.path.exists(args.test_dir):
        os.mkdir(args.test_dir)
    if not os.path.exists(os.path.join("data", "split")):
        os.mkdir(os.path.join("data", "split"))

    with open(os.path.join("data", "split", "train_utts"), "w", encoding="utf8") as f:
        for utt in train_utts:
            f.write(utt + "\n")
    with open(os.path.join("data", "split", "dev_utts"), "w", encoding="utf8") as f:
        for utt in dev_utts:
            f.write(utt + "\n")
    with open(os.path.join("data", "split", "test_utts"), "w", encoding="utf8") as f:
        for utt in test_utts:
            f.write(utt + "\n")

    subprocess.run(
        [
            "utils/subset_data_dir.sh",
            "--utt-list",
            os.path.join("data", "split", "train_utts"),
            args.data_dir,
            args.train_dir,
        ],
        check=True,
    )
    subprocess.run(
        [
            "utils/subset_data_dir.sh",
            "--utt-list",
            os.path.join("data", "split", "dev_utts"),
            args.data_dir,
            args.dev_dir,
        ],
        check=True,
    )
    subprocess.run(
        [
            "utils/subset_data_dir.sh",
            "--utt-list",
            os.path.join("data", "split", "test_utts"),
            args.data_dir,
            args.test_dir,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
