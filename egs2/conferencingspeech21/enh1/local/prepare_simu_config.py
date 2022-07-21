#!/usr/bin/env python

# Copyright 2021  Shanghai Jiao Tong University (Authors: Wangyou Zhang)
# Apache 2.0
import argparse
from pathlib import Path


def construct_path_dict(wav_list):
    path_dict = {}
    with wav_list.open("r") as f:
        for wavpath in f:
            wavpath = wavpath.strip()
            if not wavpath:
                continue
            wavname = Path(wavpath).expanduser().resolve().name
            path_dict[wavname] = wavpath
    return path_dict


def prepare_config(args):
    config = Path(args.config).expanduser().resolve()
    clean_list = Path(args.clean_list).expanduser().resolve()
    noise_list = Path(args.noise_list).expanduser().resolve()
    rir_list = Path(args.rir_list).expanduser().resolve()
    outfile = Path(args.outfile).expanduser().resolve()

    speech_data = construct_path_dict(clean_list)
    noise_data = construct_path_dict(noise_list)
    rir_data = construct_path_dict(rir_list)

    lines = []
    with config.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            path_clean, start_time, path_noise, path_rir, snr, scale = line.split()
            path_clean = speech_data[Path(path_clean).name]
            path_noise = noise_data[Path(path_noise).name]
            path_rir = rir_data[Path(path_rir).name]
            lines.append(
                f"{path_clean} {start_time} {path_noise} {path_rir} {snr} {scale}\n"
            )

    with outfile.open("w") as out:
        for line in lines:
            out.write(line)


def get_parser():
    """Argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", type=str, help="Path to the config file for simulation",
    )
    parser.add_argument(
        "--clean_list",
        type=str,
        required=True,
        help="Path to the list of clean speech audio file for simulation",
    )
    parser.add_argument(
        "--noise_list",
        type=str,
        required=True,
        help="Path to the list of noise audio file for simulation",
    )
    parser.add_argument(
        "--rir_list",
        type=str,
        required=True,
        help="Path to the list of RIR audio file for simulation",
    )
    parser.add_argument("--outfile", type=str, required=True)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    prepare_config(args)
