#!/usr/bin/env python

# Copyright 2021  Shanghai Jiao Tong University (Authors: Jing Shi)
# Apache 2.0
import argparse
from pathlib import Path

from espnet2.fileio.datadir_writer import DatadirWriter


def prepare_data(args):
    audiodirs = [Path(audiodir).expanduser().resolve() for audiodir in args.audiodirs]
    if args.uttid_prefix:
        audios = {
            "_".join([args.uttid_prefix, str(path.parent.stem), str(path.stem)]): str(
                path
            )
            for audiodir in audiodirs
            for path in audiodir.rglob("*.wav")
        }
    else:
        audios = {
            "_".join([path.parent, path.stem]): str(path)
            for audiodir in audiodirs
            for path in audiodir.rglob("*.wav")
        }
    with DatadirWriter(args.outdir) as writer:
        for uttid, utt_path in audios.items():
            writer["wav.scp"][uttid] = utt_path
            writer["spk1.scp"][uttid] = utt_path
            writer["utt2spk"][uttid] = uttid


def get_parser():
    """Argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audiodirs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the directories containing simulated audio files",
    )
    parser.add_argument(
        "--uttid_prefix",
        type=str,
        default="",
        help="Prefix to be appended to each utterance ID",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Paths to the directory for storing *.scp, utt2spk, spk2utt",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    prepare_data(args)
