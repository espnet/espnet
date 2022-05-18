#!/usr/bin/env python
#  coding: utf-8

import argparse
import os
import sys
from os.path import join

import numpy as np
from kaldiio import ReadHelper


def get_parser():
    parser = argparse.ArgumentParser(
        description="Convet kaldi-style features to numpy arrays",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("scp_file", type=str, help="scp file")
    parser.add_argument("out_dir", type=str, help="output directory")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])
    os.makedirs(args.out_dir, exist_ok=True)
    with ReadHelper(f"scp:{args.scp_file}") as f:
        for utt_id, arr in f:
            out_path = join(args.out_dir, f"{utt_id}-feats.npy")
            np.save(out_path, arr, allow_pickle=False)
    sys.exit(0)
