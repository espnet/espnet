#!/usr/bin/env python

# Copyright 2021  Shanghai Jiao Tong University (Authors: Wangyou Zhang)
# Apache 2.0
import argparse
from pathlib import Path

from espnet2.utils.types import str2bool


def prepare_data(args):
    datalist = Path(args.datalist).expanduser().resolve()
    audiodirs = [Path(audiodir).expanduser() for audiodir in args.audiodirs]
    outfile = Path(args.outfile).expanduser().resolve()
    audios = {
        path.name: str(path)
        for audiodir in audiodirs
        for path in audiodir.rglob("*." + args.audio_format)
    }
    missing_files = []
    with outfile.open("w") as out, datalist.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            wavname, others = line.split(maxsplit=1)
            if args.ignore_missing_files:
                if wavname not in audios:
                    missing_files.append(wavname)
                    continue
            else:
                assert wavname in audios, "No such file %s in %s" % (
                    wavname,
                    str([str(p) for p in audiodirs]),
                )
            out.write(audios[wavname] + " " + others + "\n")
    if args.ignore_missing_files and len(missing_files) > 0:
        print(
            "{} wav missing files are skipped:\n{}".format(
                len(missing_files), "\n  ".join(missing_files)
            )
        )


def get_parser():
    """Argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "datalist", type=str, help="Path to the list of audio files for training"
    )
    parser.add_argument("--outfile", type=str, required=True)
    parser.add_argument(
        "--audiodirs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the directories containing audio files",
    )
    parser.add_argument("--audio-format", type=str, default="wav")
    parser.add_argument("--ignore-missing-files", type=str2bool, default=False)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    prepare_data(args)
