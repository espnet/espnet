#!/usr/bin/env python3

# Copyright 2019 Ryuichi Yamamoto
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import os
import sys

from nnmnkwii.io import hts


def get_parser():
    parser = argparse.ArgumentParser(
        description="Prepare segments from HTS-style alignment files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("wav_scp", type=str, help="wav scp file")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])

    with open(args.wav_scp) as f:
        for line in f:
            recording_id, path = line.split()
            lab_path = path.replace("wav/", "lab/").replace(".wav", ".lab")
            assert os.path.exists(lab_path)

            labels = hts.load(lab_path)
            assert "sil" in labels[0][-1]
            assert "sil" in labels[-1][-1]
            segment_begin = "{:.3f}".format(labels[0][1] * 1e-7)
            segment_end = "{:.3f}".format(labels[-1][0] * 1e-7)

            # As we assume that there's only a single utterance per recording,
            # utt_id is same as recording_id.
            # https://kaldi-asr.org/doc/data_prep.html
            utt_id = recording_id
            sys.stdout.write(
                "{} {} {} {}\n".format(utt_id, recording_id, segment_begin, segment_end)
            )
