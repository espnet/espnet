#!/usr/bin/env python3

# Copyright 2021 Carnegie Mellon University (Yifan Peng)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# References:
# https://www.esat.kuleuven.be/psi/spraak/downloads/
# https://arxiv.org/pdf/1805.02922.pdf
# https://arxiv.org/pdf/2008.01994.pdf (for train/dev/test split)


import argparse
import glob
import os
import random
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser(description="Process Grabo dataset.")
parser.add_argument(
    "--data_path",
    type=str,
    default="downloads/grabo/grabo/speakers",
    help="folder containing the original data",
)
parser.add_argument("--sox_path", type=str, help="path to sox command")
parser.add_argument(
    "--train_dir",
    type=str,
    default="data/train",
    help="output folder for training data",
)
parser.add_argument(
    "--dev_dir", type=str, default="data/dev", help="output folder for validation data"
)
parser.add_argument(
    "--test_dir", type=str, default="data/test", help="output folder for test data"
)
args = parser.parse_args()


def frametotask(infile):
    """Adapted from the originial dataset.
    infile has an extension of .xml
    """

    semantic = dict()
    # read the frame xml file
    root = ET.parse(infile).getroot()
    semantic["name"] = root[0].text.strip()
    semantic["args"] = dict()
    if len(root) >= 2:
        for arg in root[1]:
            if arg.text is not None:
                semantic["args"][arg.tag] = arg.text.strip()
            else:
                semantic["args"][arg.tag] = ""

    # create the root element
    root = ET.Element(semantic["name"], attrib=semantic["args"])

    return ET.tostring(root).decode("ascii")


# Generate train/dev/test sets
# For each speaker, we randomly select 2 recordings of each command for training,
# 4 for validation, and use the remaining 9 recordings for testing.
# Reference: https://arxiv.org/pdf/2008.01994.pdf
random.seed(2021)
processed_dict = {"train": [], "dev": [], "test": []}
speaker_list = os.listdir(args.data_path)  # relative paths, or speaker names: 'pp2'
for spk in speaker_list:
    command_list = os.listdir(
        os.path.join(args.data_path, spk, "spchdatadir")
    )  # relative paths, e.g. 'recording1'
    for cmd in command_list:
        wav_list = os.listdir(
            os.path.join(args.data_path, spk, "spchdatadir", cmd)
        )  # relative paths, e.g. 'Voice_1.wav'
        wav_list.sort()
        random.shuffle(wav_list)
        random.shuffle(wav_list)
        wav_dict = {"train": wav_list[:2], "dev": wav_list[2:6], "test": wav_list[6:]}
        for n in ["train", "dev", "test"]:
            for wav in wav_dict[n]:
                wav_abspath = os.path.abspath(
                    os.path.join(args.data_path, spk, "spchdatadir", cmd, wav)
                )
                wav_id = f'{spk}_{cmd}_{wav.rstrip(".wav")}'
                task_str = frametotask(
                    os.path.join(
                        args.data_path,
                        spk,
                        "framedir",
                        cmd,
                        wav.rstrip(".wav") + ".xml",
                    )
                )
                task_str = "-".join(task_str.split())
                processed_dict[n].append(
                    {"wav_id": wav_id, "wav_abspath": wav_abspath, "task_str": task_str}
                )

# Write data into text, wav.scp, utt2spk
for n in ["train", "dev", "test"]:
    parent_dir = getattr(args, f"{n}_dir")
    sample_list = processed_dict[n]

    with open(os.path.join(parent_dir, "text"), "w") as text_f, open(
        os.path.join(parent_dir, "wav.scp"), "w"
    ) as wav_scp_f, open(os.path.join(parent_dir, "utt2spk"), "w") as utt2spk_f:
        for sample in sample_list:
            text_f.write(sample["wav_id"] + " " + sample["task_str"] + "\n")
            downsampled_wav = (
                f'{args.sox_path} {sample["wav_abspath"]} -t wav -r 16k -c 1 - |'
            )
            wav_scp_f.write(sample["wav_id"] + " " + downsampled_wav + "\n")
            utt2spk_f.write(sample["wav_id"] + " " + sample["wav_id"] + "\n")
