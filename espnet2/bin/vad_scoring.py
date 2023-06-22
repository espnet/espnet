#!/usr/bin/env python3
import argparse
import sys
from typing import List, Union

import numpy as np
from typeguard import check_argument_types

from espnet2.utils import config_argparse
from espnet.utils.cli_utils import get_commandline_args


def scoring(
    hyp_file: str,
    ref_file: str,
    output_file: Union[int, str],
):
    assert check_argument_types()

    lines_hyp = open(hyp_file).readlines()
    lines_ref = open(ref_file).readlines()

    # calculate the total P and R given the time stamp
    total_TP, total_FP, total_FN = 0, 0, 0

    for i in range(len(lines_ref)):
        time_stamp_ref = " ".join(lines_ref[i].split(" ")[1:])
        time_stamp_hyp = " ".join(lines_hyp[i].split(" ")[1:])
        ref_array = np.zeros(1000)
        hyp_array = np.zeros(1000)

        ref_time_stamps = [int(float(item) * 100) for item in time_stamp_ref.split(" ")]

        for j in range(0, len(ref_time_stamps), 2):
            ref_array[ref_time_stamps[j] : ref_time_stamps[j + 1]] = 1

        try:
            hyp_time_stamps = [
                int(float(item) * 100) for item in time_stamp_hyp.split(" ")
            ]
        except (ValueError, TypeError, AttributeError):
            print("Unable to process time stamps!")
            print(lines_hyp[i].strip())
            continue

        if len(hyp_time_stamps) % 2 != 0:
            print("start without an end!")
            print(hyp_time_stamps)
            continue

        for j in range(0, len(hyp_time_stamps), 2):
            hyp_array[hyp_time_stamps[j] : hyp_time_stamps[j + 1]] = 1

        # calculate TP, FP, FN
        TP = np.sum((ref_array[:1000] == 1) & (hyp_array[:1000] == 1))
        FP = np.sum((ref_array[:1000] == 0) & (hyp_array[:1000] == 1))
        FN = np.sum((ref_array[:1000] == 1) & (hyp_array[:1000] == 0))

        total_TP += TP
        total_FP += FP
        total_FN += FN

    P = total_TP / (total_TP + total_FP)
    R = total_TP / (total_TP + total_FN)
    F = 2 * P * R / (P + R)
    write_file = open(output_file, "w")
    write_file.write("|Precision: " + str(round(P, 4)) + "|\n")
    write_file.write("|Recall: " + str(round(R, 4)) + "|\n")
    write_file.write("|F1_score: " + str(round(F, 4)) + "|\n")


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="VAD inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.
    parser.add_argument("--hyp_file", type=str, required=True)
    parser.add_argument("--ref_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    scoring(**kwargs)


if __name__ == "__main__":
    main()
