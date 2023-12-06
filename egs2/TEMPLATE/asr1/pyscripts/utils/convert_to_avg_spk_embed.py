#!/usr/bin/env python

import argparse
import os
import shutil
import sys


def get_parser():
    parser = argparse.ArgumentParser(
        description="""
    Replaces spk_embeds in a specified spk_embed directory with the average spk_embed
    for a given speaker.
    The spk_embeds generally reside in
        dump/${spk_embed_tag}/<data_subset>/${spk_embed_tag}.scp, whereas
    speaker-averaged spk_embeds reside in
        dump/${spk_embed_tag}/<data_subset>/spk_${spk_embed_tag}.scp.

    The old spk_embed.scp file will be renamed to spk_embed.scp.bak and
    the corresponding .ark files are left unchanged.
    If no speaker id is provided, the average spk_embed for the speaker who
    the utterance belongs to will be used in each case.

    At inference time in a TTS task, you are unlikely to have the spk_embed
    for that sentence in particular. Thus, using average spk_embeds
    during training may yield better performance at inference time.

    This is also useful for conditioning inference on a particular speaker.

    To transform the training data, this script should be run after
    spk_embeds are extracted (stage 3), but before training commences (stage 7).
    """
    )
    parser.add_argument(
        "--utt-embed-path",
        type=str,
        required=True,
        help="Path to the spk_embed file to be modified.",
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
        "--spk-embed-path",
        type=str,
        required=True,
        help="The path to the spk_{spk_embed_tag}.scp for the speakers being used.",
    )
    return parser


def check_args(args):
    utt_embed_path = args.utt_embed_path
    spk_embed_path = args.spk_embed_path
    utt2spk = args.utt2spk

    if not os.path.exists(utt_embed_path):
        sys.stderr.write(
            f"Error: provided --utt-embed-path ({utt_embed_path}) does not exist. "
        )
        sys.stderr.write("Exiting...\n")
        sys.stderr.flush()
        exit(1)

    if not os.path.exists(spk_embed_path):
        sys.stderr.write(
            f"Error: provided --spk-embed-path ({spk_embed_path}) does not exist. "
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
    utt_embed_path = args.utt_embed_path
    spk_embed_path = args.spk_embed_path

    print(f"Loading {spk_embed_path}...")
    spk_embed_paths = {}
    with open(spk_embed_path) as spembfile:
        for line in spembfile.readlines():
            spkid, spembpath = line.split()
            spk_embed_paths[spkid] = spembpath

    if spk_id and (spk_id not in spk_embed_paths):
        sys.stderr.write(
            f"Error: provided --spk-id: {spk_id} not present in --spk-embed-path."
        )
        sys.stderr.write("Exiting...\n")
        sys.stderr.flush()
        exit(1)

    print("Backing up utt_embed file...")
    print(os.path.dirname(utt_embed_path))
    shutil.copy(
        utt_embed_path,
        f"{os.path.dirname(utt_embed_path)}/os.path.filename(utt_embed_path).bak",
    )

    utt2spk_embed = []
    with open(args.utt_embed_path) as f:
        for line in f.readlines():
            utt, spk_embed = line.split()
            utt2spk_embed.append((utt, spk_embed_paths[spk_id]))

    with open(args.utt_embed_path, "w") as f:
        for utt, spk_embed in utt2spk_embed:
            f.write(f"{utt} {spk_embed}\n")
