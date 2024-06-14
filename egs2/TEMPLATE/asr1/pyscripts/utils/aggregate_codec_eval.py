#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Aggregate results."""

import argparse
import json
import logging

from tqdm import tqdm


def get_parser() -> argparse.Namespace:
    """Get parser of aggregate results."""
    parser = argparse.ArgumentParser(
        description="Aggregate results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--logdir",
        type=str,
        required=True,
        help="Input log directory.",
    )
    parser.add_argument(
        "--scoredir",
        type=str,
        required=True,
        help="Output scoring directory.",
    )
    parser.add_argument(
        "--nj",
        type=int,
        required=True,
        help="Number of sub jobs",
    )
    return parser


def aggregate_results(logdir: str, scoredir: str, nj: int) -> None:
    """Aggregate results."""
    logging.info("Aggregating results...")
    score_info = []
    for i in range(nj):
        with open("{}/result.{}.txt".format(logdir, i + 1), "r") as f:
            for line in f:
                line = line.strip().replace("'", '"').replace("inf", "Infinity")
                score_info.append(json.loads(line))
    with open("{}/utt_result.txt".format(scoredir), "w") as f, open(
        "{}/avg_result.txt".format(scoredir), "w"
    ) as f2:
        for info in tqdm(score_info):
            f.write("{}\n".format(info))
        for key in score_info[0].keys():
            if key == "key":
                continue
            avg = sum([info[key] for info in score_info]) / len(score_info)
            f2.write("{}: {}\n".format(key, avg))

    logging.info("Done.")


def main() -> None:
    """Run main function."""
    parser = get_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    aggregate_results(args.logdir, args.scoredir, args.nj)


if __name__ == "__main__":
    main()
