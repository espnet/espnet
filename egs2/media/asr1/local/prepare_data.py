#!/usr/bin/env python3

# Copyright 2022 University of Stuttgart (Pavel Denisov)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import glob
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

files = {
    "train": ["lot1", "lot2", "lot3", "lot4"],
    "dev": ["testHC_a_blanc"],
    "test": ["testHC"],
}

channel = {"l": 1, "r": 2}

idir1 = sys.argv[1]  # annotation
idir2 = sys.argv[2]  # speech
odir = sys.argv[3]

wavs = {}

for wav_file in glob.glob(
    os.path.join(idir2, "MEDIA1FR_0*", "MEDIA1FR", "BLOCK*", "*.wav")
):
    wav_id = wav_file.split("/")[-1][:-4]
    wavs[wav_id] = wav_file

data_dir = os.path.join(idir1, "MEDIA1FR_00", "MEDIA1FR", "DATA")

for subset in files.keys():
    subset_dir = os.path.join(odir, subset)
    os.makedirs(subset_dir, exist_ok=True)

    with open(os.path.join(subset_dir, "wav.scp"), "w") as wavscp, open(
        os.path.join(subset_dir, "utt2spk"), "w"
    ) as utt2spk, open(os.path.join(subset_dir, "text"), "w", encoding="utf-8") as text:
        for part in files[subset]:
            xml_root = ET.parse(os.path.join(data_dir, f"media_{part}.xml")).getroot()
            for turn in xml_root.findall(".//turn[@speaker='spk']"):
                turn_id = turn.get("id")

                slots = []

                for sem in turn.findall("./semAnnotation[@withContext='false']/sem"):
                    # xpath sem[@mode!='null'] does not work for some reason
                    if sem.get("mode") == "null":
                        continue
                    specifier = sem.get("specif")
                    if specifier == "Relative-reservation":
                        specifier = "-relative-reservation"
                    concept = sem.get("concept") + specifier
                    value = sem.get("value")
                    if value == "":
                        slots.append(concept)
                    else:
                        slots.append(f"{concept} FILL {value}")

                transcription = "".join(
                    [x.strip() for x in turn.find("./transcription").itertext()]
                )
                transcription = transcription.replace("(", "")
                transcription = transcription.replace(")", "")
                transcription = transcription.replace("*", "")

                if transcription == "":
                    continue

                text.write(f"{turn_id} {' SEP '.join(slots + [transcription])}\n")

                audio = turn.get("audioFilename").split("/")[-1].split(".")
                start = float(turn.get("startTime"))
                end = float(turn.get("endTime"))
                dur = end - start

                wavscp.write(
                    f"{turn_id} sox {wavs[audio[0]]} "
                    + "-r 16k -t wav -c 1 -b 16 -e signed - "
                    + f"trim {start} {dur} remix {channel[audio[1]]} |\n"
                )
                utt2spk.write(f"{turn_id} {turn_id}\n")

    subprocess.call("utils/fix_data_dir.sh {}".format(subset_dir), shell=True)
