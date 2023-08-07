#!/usr/bin/env python

import argparse
import os
import shutil
import sys


def get_parser():
    parser = argparse.ArgumentParser(
        description="""
    Replaces xvectors in a specified xvector directory with the average xvector
    for a given speaker.
    The xvectors generally reside in dump/xvector/<data_subset>/xvector.scp, whereas
    speaker-averaged xvectors reside in dump/xvector/<data_subset>/spk_xvector.scp.

    The old xvector.scp file will be renamed to xvector.scp.bak and
    the corresponding .ark files are left unchanged.
    If no speaker id is provided, the average xvector for the speaker who
    the utterance belongs to will be used in each case.

    At inference time in a TTS task, you are unlikely to have the xvector
    for that sentence in particular. Thus, using average xvectors
    during training may yield better performance at inference time.

    This is also useful for conditioning inference on a particular speaker.

    To transform the training data, this script should be run after
    xvectors are extracted (stage 2), but before training commences (stage 6).
    """
    )
    parser.add_argument(
        "--xvector-path",
        type=str,
        required=True,
        help="Path to the xvector file to be modified.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--spk-id",
        type=str,
        help="The id of the speaker whose average spk_id should be used",
    )
    group.add_argument(
        "--utt2spk",
        type=str,
        help="Path to the relevant utt2spk file, if the source speakers are used",
    )
    parser.add_argument(
        "--spk-xvector-path",
        type=str,
        required=True,
        help="The path to the spk_xvector.scp file for the speakers being used.",
    )
    return parser


def check_args(args):
    xvector_path = args.xvector_path
    spk_xvector_path = args.spk_xvector_path
    utt2spk = args.utt2spk

    if not os.path.exists(xvector_path):
        sys.stderr.write(
            f"Error: provided --xvector-path ({xvector_path}) does not exist. "
        )
        sys.stderr.write("Exiting...\n")
        sys.stderr.flush()
        exit(1)

    if not os.path.exists(spk_xvector_path):
        sys.stderr.write(
            f"Error: provided --spk-xvector-path ({spk_xvector_path}) does not exist. "
        )
        sys.stderr.write("Exiting...\n")
        sys.stderr.flush()
        exit(1)

    if utt2spk and (not os.path.exists(utt2spk)):
        sys.stderr.write(f"Error: provided --utt2spk file ({utt2spk}) does not exist. ")
        sys.stderr.write("Exiting...\n")
        sys.stderr.flush()
        exit(1)


if __name__ == "__main__":
    args = get_parser().parse_args()
    check_args(args)
    spk_id = args.spk_id
    utt2spk = args.utt2spk
    xvector_path = args.xvector_path
    spk_xvector_path = args.spk_xvector_path

    print(f"Loading {spk_xvector_path}...")
    spk_xvector_paths = {}
    with open(spk_xvector_path) as spembfile:
        for line in spembfile.readlines():
            spkid, spembpath = line.split()
            spk_xvector_paths[spkid] = spembpath

    if spk_id and (spk_id not in spk_xvector_paths):
        sys.stderr.write(
            f"Error: provided --spk-id: {spk_id} not present in --spk-xvector-path."
        )
        sys.stderr.write("Exiting...\n")
        sys.stderr.flush()
        exit(1)

    print("Backing up xvector file...")
    print(os.path.dirname(xvector_path))
    shutil.copy(xvector_path, f"{os.path.dirname(xvector_path)}/xvector.scp.bak")

    utt2xvector = []
    with open(args.xvector_path) as f:
        for line in f.readlines():
            utt, xvector = line.split()
            utt2xvector.append((utt, spk_xvector_paths[spk_id]))

    with open(args.xvector_path, "w") as f:
        for utt, xvector in utt2xvector:
            f.write(f"{utt} {xvector}\n")
