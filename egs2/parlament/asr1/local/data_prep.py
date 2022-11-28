#!/usr/bin/env python3

# Copyright 2016  Allen Guo

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

import os
import re
import sys

if len(sys.argv) != 3:
    print("Usage: python data_prep.py [parla_root] [sph2pipe]")
    sys.exit(1)
parla_root = sys.argv[1]
sph2pipe = sys.argv[2]

sph_dir = {"train": "parla_clstk", "test": "parlatest_clstk"}

for x in ["train", "test"]:
    with open(os.path.join(parla_root, "clean_" + x + ".tsv")) as transcript_f, open(
        os.path.join("data", x, "text"), "w"
    ) as text_f, open(os.path.join("data", x, "wav.scp"), "w") as wav_scp_f, open(
        os.path.join("data", x, "utt2spk"), "w"
    ) as utt2spk_f:

        text_f.truncate()
        wav_scp_f.truncate()
        utt2spk_f.truncate()

        lines = transcript_f.readlines()
        final_lines = []

        for line in lines:
            split_data = line.split("\t")
            utt_id = split_data[1].split("/")[-1].replace(".wav", "")
            fin_string = utt_id + line
            final_lines.append(fin_string)

        lines = sorted(final_lines, key=lambda s: s.split("\t")[0])

        for line in lines:
            split_data = line.split("\t")

            if "clean_train" not in split_data[1] and "clean_test" not in split_data[1]:
                continue

            utt_id = split_data[0]

            utt2spk_f.write(utt_id + " " + split_data[0] + "\n")

            text_f.write(utt_id + " " + split_data[2] + "\n")

            wav_scp_f.write(
                utt_id + " " + os.path.join(parla_root, split_data[1]) + "\n"
            )
