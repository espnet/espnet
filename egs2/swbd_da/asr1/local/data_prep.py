#!/usr/bin/env python3

# Copyright 2021 University of Stuttgart (Pavel Denisov)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import glob
import os
import sys
import xml.etree.ElementTree as ET

if len(sys.argv) != 3:
    print("Usage: python data_prep.py <LDC97S62 path> <LDC2009T26 path>")
    sys.exit(1)

audio_root = sys.argv[1]
nxt_root = os.path.join(sys.argv[2], "nxt_switchboard_ann", "xml")

channel = {"A": 1, "B": 2}
speaker = {}

corpus_resources_root = ET.parse(
    os.path.join(nxt_root, "corpus-resources", "dialogues.xml")
).getroot()
for dialogue in corpus_resources_root.findall(".//dialogue"):
    dialogue_id = "sw" + dialogue.attrib["swbdid"]
    speaker[dialogue_id] = {}
    for pointer in dialogue.findall(".//{http://nite.sourceforge.net/}pointer"):
        speaker[dialogue_id][pointer.attrib["role"]] = pointer.attrib["href"].split(
            "#"
        )[1][3:-1]

sph = {}

for sph_file in glob.glob(os.path.join(audio_root, "*/swb1/sw*.sph")):
    dialogue_id = sph_file[-10:-4]
    sph[dialogue_id] = sph_file

# Data splits local/{train,valid,test}.lst
# from the paper: Ji Young Lee*, Franck Dernoncourt*.
# Sequential Short-Text Classification with Recurrent and Convolutional Neural Networks.
# NAACL 2016. (* indicates equal contribution)

for subset in ["train", "valid", "test"]:
    with open(os.path.join("data", subset, "text"), "w") as text_f, open(
        os.path.join("data", subset, "wav.scp"), "w"
    ) as wav_scp_f, open(
        os.path.join("data", subset, "utt2spk"), "w"
    ) as utt2spk_f, open(
        os.path.join("local", subset + ".lst")
    ) as dialogues_f:
        for line in dialogues_f:
            dialogue_id = line.strip()

            for role in ["A", "B"]:
                terminals = {}

                terminals_file = os.path.join(
                    nxt_root, "terminals", f"{dialogue_id}.{role}.terminals.xml"
                )

                if not os.path.exists(terminals_file):
                    continue

                terminals_root = ET.parse(terminals_file).getroot()

                for terminal in terminals_root.findall(".//word"):
                    start_str = terminal.attrib["{http://nite.sourceforge.net/}start"]
                    end_str = terminal.attrib["{http://nite.sourceforge.net/}end"]
                    if (
                        start_str != "non-aligned"
                        and start_str != "n/a"
                        and end_str != "n/a"
                    ):
                        terminals[
                            terminal.attrib["{http://nite.sourceforge.net/}id"]
                        ] = {
                            "start": float(start_str),
                            "end": float(end_str),
                        }

                dial_act_root = ET.parse(
                    os.path.join(
                        nxt_root, "dialAct", f"{dialogue_id}.{role}.dialAct.xml"
                    )
                ).getroot()

                for dial_act in dial_act_root.findall(".//da"):
                    words = dial_act.attrib["niteType"]
                    if words == "excluded":
                        continue

                    utt_id = (
                        speaker[dialogue_id][role]
                        + "_"
                        + dialogue_id
                        + "_"
                        + dial_act.attrib["{http://nite.sourceforge.net/}id"][2:]
                    )

                    dial_act_children = dial_act.findall(
                        ".//{http://nite.sourceforge.net/}child"
                    )

                    start_terminal_id = (
                        dial_act_children[0].attrib["href"].split("#")[1][3:-1]
                    )
                    end_terminal_id = (
                        dial_act_children[-1].attrib["href"].split("#")[1][3:-1]
                    )

                    if (
                        start_terminal_id not in terminals
                        or end_terminal_id not in terminals
                    ):
                        continue

                    start = terminals[start_terminal_id]["start"]
                    end = terminals[end_terminal_id]["end"]

                    text_f.write(utt_id + " " + words + "\n")

                    wav_scp_f.write(
                        "{} sox {} -r 16k -t wav -c 1 -b 16 -e signed - ".format(
                            utt_id, sph[dialogue_id]
                        )
                        + "trim {} {} remix {} |\n".format(
                            start, end - start, channel[role]
                        )
                    )
                    utt2spk_f.write(utt_id + " " + speaker[dialogue_id][role] + "\n")
