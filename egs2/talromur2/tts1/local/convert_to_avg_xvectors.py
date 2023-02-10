#!/usr/bin/env python

import argparse
import os
import shutil
import sys

parser = argparse.ArgumentParser(
    description="""
Replaces xvectors in a specified xvector directory with the average xvector
for a given speaker. The xvectors generally reside in
dump/xvector/<data_subset>/xvector.scp, whereas speaker-averaged xvectors
reside in dump/xvector/<data_subset>/spk_xvector.scp. Since at inference time,
you are unlikely to have the xvector for that sentence in particular,
using average xvectors should be preferred.
The old xvector.scp file will be moved to xvector.scp.backup and
the corresponding .ark files are left unchanged.
If no speaker id is provided, the average xvector for the speaker who
the utterance belongs to will be used in each case.
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

args = parser.parse_args()

xvector_dir = args.xvector_path

if not os.path.exists(args.xvector_path):
    sys.stderr.write(
        f"Error: provided --xvector-path ({args.xvector_path}) does not exist."
    )
    sys.stderr.write(" Exiting...\n")
    sys.stderr.flush()
    exit(1)

if not os.path.exists(args.spk_xvector_path):
    sys.stderr.write(
        f"Error: provided --spk-xvector-path ({args.spk_xvector_path}) does not exist."
    )
    sys.stderr.write(" Exiting...\n")
    sys.stderr.flush()
    exit(1)

if args.utt2spk and (not os.path.exists(args.utt2spk)):
    sys.stderr.write(f"Error: provided --utt2spk file ({args.utt2spk}) does not exist.")
    sys.stderr.write(" Exiting...\n")
    sys.stderr.flush()
    exit(1)

print(f"Loading {args.spk_xvector_path}...")
spk_xvector_paths = {}
with open(args.spk_xvector_path) as spembfile:
    for line in spembfile.readlines():
        spkid, spembpath = line.split()
        spk_xvector_paths[spkid] = spembpath


if args.spk_id and (args.spk_id not in spk_xvector_paths):
    sys.stderr.write(
        f"Error: --spk-id:{args.spk_id} not present in provided --spk-xvector-path."
    )
    sys.stderr.write(" Exiting...\n")
    sys.stderr.flush()
    exit(1)

print("Backing up xvector file...")
print(os.path.dirname(args.xvector_path))
shutil.copy(
    args.xvector_path, f"{os.path.dirname(args.xvector_path)}/xvector.scp.backup"
)

utt2xvector = []
with open(args.xvector_path) as f:
    for line in f.readlines():
        utt, xvector = line.split()
        utt2xvector.append((utt, spk_xvector_paths[args.spk_id]))

with open(args.xvector_path, "w") as f:
    for utt, xvector in utt2xvector:
        f.write(f"{utt} {xvector}\n")
