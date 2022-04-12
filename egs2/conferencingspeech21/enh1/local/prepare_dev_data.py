#!/usr/bin/env python

# Copyright 2021  Shanghai Jiao Tong University (Authors: Wangyou Zhang)
# Apache 2.0
import argparse
from pathlib import Path
import re

from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.utils.types import str2bool


def prepare_data(args):
    config_file = Path(args.config_file).expanduser().resolve()
    audiodirs = [Path(audiodir).expanduser().resolve() for audiodir in args.audiodirs]
    audios = {
        path.stem: str(path)
        for audiodir in audiodirs
        for path in audiodir.rglob("*.wav")
    }
    suffix = "_" + args.uttid_suffix if args.uttid_suffix else ""
    with DatadirWriter(args.outdir) as writer, config_file.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            path_clean, start_time, path_noise, path_rir, snr, scale = line.split()
            uttid = "#".join(
                [
                    Path(path_clean).stem,
                    Path(path_noise).stem,
                    Path(path_rir).stem,
                    start_time,
                    snr,
                    scale,
                ]
            )
            writer["wav.scp"][uttid + suffix] = audios[uttid]
            if args.use_reverb_ref:
                repl = r"/reverb_ref/\1"
            else:
                repl = r"/noreverb_ref/\1"
            writer["spk1.scp"][uttid + suffix] = re.sub(
                r"/mix/([^\\]+\.wav$)", repl, audios[uttid]
            )
            if "librispeech" in path_clean:
                spkid = "-".join(path_clean.split("/")[-3:-1])
            else:
                spkid = path_clean.split("/")[-2]
            writer["utt2spk"][uttid + suffix] = spkid


def get_parser():
    """Argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=str, help="Path to the list of audio files for training"
    )
    parser.add_argument(
        "--audiodirs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the directories containing simulated audio files",
    )
    parser.add_argument(
        "--uttid_suffix",
        type=str,
        default="",
        help="suffix to be appended to each utterance ID",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Paths to the directory for storing *.scp, utt2spk, spk2utt",
    )
    parser.add_argument(
        "--use_reverb_ref",
        type=str2bool,
        default=True,
        help="True to use reverberant references, False to use non-reverberant ones",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    prepare_data(args)
