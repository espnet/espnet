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
            wavname = Path(wavpath).expanduser().resolve().with_suffix("").name
            path_dict[wavname] = wavpath
    return path_dict


def prepare_config(args):
    audiodir = Path(args.audiodir).expanduser()
    clean_list = Path(args.clean_list).expanduser().resolve()
    noise_list = Path(args.noise_list).expanduser().resolve()
    outfile = Path(args.outfile).expanduser().resolve()

    speech_data = construct_path_dict(clean_list)
    noise_data = construct_path_dict(noise_list)
    audios = {
        folder: {
            path.with_suffix("").name: str(path)
            for path in (audiodir / folder).rglob("*." + args.audio_format)
        }
        for folder in ("mix", "noreverb_ref", "reverb_ref")
    }
    keys = audios["mix"].keys()
    assert keys == audios["noreverb_ref"].keys() == audios["reverb_ref"].keys()

    with outfile.open("w") as out:
        for name in keys:
            path_clean, path_noise, path_rir, start_time, snr, scale = name.split("#")
            path_clean = speech_data[path_clean]
            path_noise = noise_data[path_noise]
            out.write(
                f"{path_clean} {start_time} {path_noise} "
                f"/path/{args.tag}/{path_rir}.wav {snr} {scale}\n"
            )


def get_parser():
    """Argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audiodir",
        type=str,
        required=True,
        help="Paths to the directory containing simulated audio files",
    )
    parser.add_argument("--audio-format", type=str, default="wav")
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
    parser.add_argument("--outfile", type=str, required=True)
    parser.add_argument("--tag", type=str, default="linear")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    prepare_config(args)
