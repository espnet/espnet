#!/usr/bin/env python3

# Copyright 2021 Takaaki Saeki
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import glob
import os

import soundfile as sf
import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_thresh", type=float, default=-0.5, required=False)
    parser.add_argument(
        "--db_raw", type=str, default="downloads/jtuberaw", required=False
    )
    parser.add_argument(
        "--db_split", type=str, default="downloads/jtubesplit", required=False
    )
    args = parser.parse_args()

    thresh_ctc = args.score_thresh

    raw_dir = os.path.join(os.path.dirname(__file__), "../{}".format(args.db_raw))
    data_dir = os.path.join(os.path.dirname(__file__), "../{}".format(args.db_split))
    ctcscore_path = os.path.join(raw_dir, "ctcscore.txt")
    raw_transcript_path = os.path.join(data_dir, "transcript_raw.txt")
    pruned_transcript_path = os.path.join(data_dir, "transcript_prune.txt")
    if os.path.exists(pruned_transcript_path):
        os.remove(pruned_transcript_path)

    d_ctcscore = dict()
    with open(ctcscore_path, "r") as fr:
        for line in fr:
            line_list = line.strip().split(" ", 5)
            d_ctcscore[line_list[0]] = float(line_list[4])

    with open(raw_transcript_path, "r") as fr:
        for line in fr:
            line_list = line.strip().split(" ", 1)
            stem = line_list[0]
            try:
                if d_ctcscore[stem] > thresh_ctc:
                    with open(pruned_transcript_path, "a") as fa:
                        fa.write(line)
            except KeyError:
                pass
