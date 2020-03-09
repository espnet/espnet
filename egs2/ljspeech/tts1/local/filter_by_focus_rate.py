#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#           2020 Songxiang Liu
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Filter samples by focus rates."""

import argparse
import logging
import os

import kaldiio


def main():
    """Run filtering by focus rate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--focus-rates-scp", type=str,
                        help="scp file of focus rates")
    parser.add_argument("--durations-scp", type=str,
                        help="scp file of focus rates")
    parser.add_argument("--feats-scp", type=str,
                        help="scp file of focus rates")
    parser.add_argument("--speech-shape-scp", type=str,
                        help="scp file of feat shapes")
    parser.add_argument("--text-shape-scp", type=str,
                        help="scp file of text shapes")
    parser.add_argument("--threshold", type=float, default=None,
                        help="threshold value of focus rates (0.0, 1.0)")
    parser.add_argument("--verbose", default=1, type=int,
                        help="verbose option")
    args = parser.parse_args()

    # logging info
    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    else:
        logging.basicConfig(
            level=logging.WARN, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
        logging.warning('Skip DEBUG/INFO messages')

    # check threshold is valid
    assert args.threshold > 0 and args.threshold < 1

    # load focus rates scp
    feat_reader = kaldiio.load_scp(args.feats_scp)
    dur_reader = kaldiio.load_scp(args.durations_scp)
    fr_reader = kaldiio.load_scp(args.focus_rates_scp)
    speech_shape_reader = kaldiio.load_scp(args.speech_shape_scp)
    text_shape_reader = kaldiio.load_scp(args.text_shape_scp)

    # define writer
    dirname = os.path.dirname(args.feats_scp)
    feat_fid = open(f"{dirname}/feats_filtered.scp", "w")
    dur_fid = open(f"{dirname}/durations_filtered.scp", "w")
    shape_dirname = os.path.dirname(args.speech_shape_scp)
    speech_shape_fid = open(f"{shape_dirname}/speech_shape_filtered", "w")
    text_shape_fid = open(f"{shape_dirname}/text_shape_filtered", "w")

    # do filtering
    drop_count = 0
    for utt_id in fr_reader.keys():
        focus_rate = fr_reader[utt_id]
        if focus_rate >= args.threshold:
            feat_fid.write(f"{utt_id} {feat_reader._dict[utt_id]}\n")
            dur_fid.write(f"{utt_id} {dur_reader._dict[utt_id]}\n")
            speech_shape_fid.write(f"{utt_id} {speech_shape_reader._dict[utt_id]}\n")
            text_shape_fid.write(f"{utt_id} {text_shape_reader._dict[utt_id]}\n")
        else:
            drop_count += 1
            logging.info(f"{utt_id} is dropped (focus rate: {focus_rate}).")
    logging.info(f"{drop_count} utts are dropped by filtering.")

    # close writer instances
    feat_fid.close()
    dur_fid.close()
    speech_shape_fid.close()
    text_shape_fid.close()


if __name__ == "__main__":
    main()
