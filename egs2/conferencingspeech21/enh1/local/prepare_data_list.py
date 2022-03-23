#!/usr/bin/env python

# Copyright 2021  Shanghai Jiao Tong University (Authors: Wangyou Zhang)
# Apache 2.0
import argparse
from pathlib import Path


def prepare_data(args):
    datalist = Path(args.datalist).expanduser().resolve()
    audiodirs = [Path(audiodir).expanduser() for audiodir in args.audiodirs]
    outfile = Path(args.outfile).expanduser().resolve()
    audios = {
        path.name: str(path)
        for audiodir in audiodirs
        for path in audiodir.rglob("*." + args.audio_format)
    }
    with outfile.open("w") as out, datalist.open("r") as f:
        for wavname in f:
            wavname = wavname.strip()
            if not wavname:
                continue
            assert wavname in audios, "No such file %s in %s" % (
                wavname,
                str([str(p) for p in audiodirs]),
            )
            out.write(audios[wavname] + "\n")


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
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    prepare_data(args)
