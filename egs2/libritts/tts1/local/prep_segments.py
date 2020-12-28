#!/usr/bin/env python3

"""Prepare segments file."""

# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import os
import shutil
import sys

from distutils.util import strtobool


def load_lab(lab_path):
    """Load the force alignment labels."""
    with open(lab_path, "r") as f:
        lines = [line.replace("\n", "") for line in f.readlines()]
    start_times_in_sec = [float(line.split("\t")[0]) for line in lines]
    end_times_in_sec = [float(line.split("\t")[1]) for line in lines]
    phonemes = [line.split("\t")[2] for line in lines]
    return start_times_in_sec, end_times_in_sec, phonemes


def main():
    """Run main process."""
    parser = argparse.ArgumentParser(
        description="Prepare segments from the force alignment files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("wav_scp", type=str, help="wav scp file")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        stream=sys.stdout,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    segments_path = os.path.dirname(args.wav_scp) + "/segments"

    with open(args.wav_scp) as f, open(segments_path, "w") as g:
        for line in f:
            recording_id, path = line.split()
            lab_path = path.replace(".wav", ".lab")
            if not os.path.exists(lab_path):
                logging.warning(f"{lab_path} does not exists. skipped it.")
                continue
            labels = load_lab(lab_path)
            if labels[2][0] in ["sil"]:
                segment_begin = "{:.3f}".format(labels[0][1])
            else:
                segment_begin = "{:.3f}".format(labels[0][0])
            assert len(labels[2][-1]) == 0
            if labels[2][-2] in ["sp"]:
                segment_end = "{:.3f}".format(labels[0][-2])
            else:
                segment_end = "{:.3f}".format(labels[1][-1])

            # As we assume that there's only a single utterance per recording,
            # utt_id is same as recording_id.
            # https://kaldi-asr.org/doc/data_prep.html
            utt_id = recording_id
            g.write(
                "{} {} {} {}\n".format(utt_id, recording_id, segment_begin, segment_end)
            )


if __name__ == "__main__":
    main()
