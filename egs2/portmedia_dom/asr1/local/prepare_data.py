#!/usr/bin/env python3

# Copyright 2022 University of Stuttgart (Pavel Denisov)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import glob
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

idir = sys.argv[1]
odir = sys.argv[2]

os.makedirs(odir, exist_ok=True)

files = sorted(
    glob.glob(os.path.join(idir, "PMDOM2FR_00", "PMDOM2FR", "BLOCK*", "*.xml"))
)

subsets = {
    "train": files[0:400],
    "dev": files[400:500],
    "test": files[500:700],
}

for subset in subsets.keys():
    subset_dir = os.path.join(odir, subset)
    os.makedirs(subset_dir, exist_ok=True)

    with open(os.path.join(subset_dir, "wav.scp"), "w") as wavscp, open(
        os.path.join(subset_dir, "utt2spk"), "w"
    ) as utt2spk, open(os.path.join(subset_dir, "text"), "w", encoding="utf-8") as text:
        for file in subsets[subset]:
            file_id = file.split("/")[-1][:-4]
            turn_id = 0

            # annotation for the wrong audio
            if file_id == "08730_887":
                continue

            xml_root = ET.parse(file).getroot()
            compere = ""

            for speaker in xml_root.findall("./Speakers/Speaker"):
                if speaker.get("name").lower().startswith("compère"):
                    compere = speaker.get("id")
                    break

            for turn in xml_root.findall(".//Turn"):
                turn_id += 1

                if turn.get("speaker") == compere:
                    continue

                transcription = " ".join(sum([t.split() for t in turn.itertext()], []))
                transcription = transcription.replace("(", "")
                transcription = transcription.replace(")", "")
                transcription = transcription.replace("*", "")

                if transcription == "":
                    continue

                slots = [
                    e.get("concept") + " FILL " + e.get("valeur")
                    for e in turn.findall("./SemDebut")
                    if e.get("concept") != "null"
                ]

                utt = f"{file_id}_{turn_id}"

                text.write(f"{utt} {' SEP '.join(slots + [transcription])}\n")

                start = float(turn.get("startTime"))
                end = float(turn.get("endTime"))
                dur = end - start

                if dur == 0.0:
                    continue

                wavscp.write(
                    f"{utt} sox {file[:-4]}.wav "
                    + "-r 16k -t wav -c 1 -b 16 -e signed - "
                    + f"trim {start} {dur} remix 1 |\n"
                )
                utt2spk.write(f"{utt} {utt}\n")

    subprocess.call("utils/fix_data_dir.sh {}".format(subset_dir), shell=True)
