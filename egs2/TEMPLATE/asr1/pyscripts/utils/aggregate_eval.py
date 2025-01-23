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
                line = (
                    line.strip()
                    .replace("'", '"')
                    .replace("inf", "Infinity")
                    .replace(f"nan", "0.0")
                )
                score_info.append(json.loads(line))
    # score cer, wer
    cer_wer_score = {}
    with open("{}/utt_result.txt".format(scoredir), "w") as f, open(
        "{}/avg_result.txt".format(scoredir), "w"
    ) as f2:
        for info in tqdm(score_info):
            f.write("{}\n".format(info))
        score_info = [
            {k: v for k, v in info.items() if k not in ["espnet_hyp_text", "ref_text"]}
            for info in score_info
        ]
        for key in score_info[0].keys():
            if key == "key" or "text" in key:
                continue
            elif "wer" in key or "cer" in key:
                f2.write(
                    "{}: {}\n".format(key, sum([info[key] for info in score_info]))
                )
                error_rate_tags = key.split("_")
                assert (
                    len(error_rate_tags) == 3
                ), "error rate must be in <tag>_<cer/wer>_<type> format"
                error_rate_tag, error_rate_type, error_type = error_rate_tags
                if error_rate_tag not in cer_wer_score:
                    cer_wer_score[error_rate_tag] = {}
                cer_wer_score[error_rate_tag][key] = sum(
                    [info[key] for info in score_info]
                )
                continue
            avg = sum([info[key] for info in score_info]) / len(score_info)
            f2.write("{}: {}\n".format(key, avg))

        # Process WER/CER
        for error_rate_tag in cer_wer_score.keys():
            error_rate_info = cer_wer_score[error_rate_tag]
            ins_num, del_num, sub_num, equ_num = (
                "{}_wer_insert".format(error_rate_tag),
                "{}_wer_delete".format(error_rate_tag),
                "{}_wer_replace".format(error_rate_tag),
                "{}_wer_equal".format(error_rate_tag),
            )
            wer = (
                error_rate_info[ins_num]
                + error_rate_info[del_num]
                + error_rate_info[sub_num]
            ) / (
                error_rate_info[sub_num]
                + error_rate_info[equ_num]
                + error_rate_info[del_num]
            )
            ins_num, del_num, sub_num, equ_num = (
                "{}_cer_insert".format(error_rate_tag),
                "{}_cer_delete".format(error_rate_tag),
                "{}_cer_replace".format(error_rate_tag),
                "{}_cer_equal".format(error_rate_tag),
            )
            cer = (
                error_rate_info[ins_num]
                + error_rate_info[del_num]
                + error_rate_info[sub_num]
            ) / (
                error_rate_info[sub_num]
                + error_rate_info[equ_num]
                + error_rate_info[del_num]
            )
            f2.write("{}_wer {}\n".format(error_rate_tag, wer))
            f2.write("{}_cer {}\n".format(error_rate_tag, cer))
    logging.info("Done.")


def main() -> None:
    """Run main function."""
    parser = get_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    aggregate_results(args.logdir, args.scoredir, args.nj)


if __name__ == "__main__":
    main()
