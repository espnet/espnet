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
import random
import re
import sys

if len(sys.argv) != 3:
    print("Usage: python data_prep.py [an4_root] [sph2pipe]")
    sys.exit(1)
an4_root = sys.argv[1]
sph2pipe = sys.argv[2]

random.seed(0)

sph_dir = {"train": "an4_clstk", "test": "an4test_clstk"}

train_langs = set()

for x in [("train_minian4", "train"), ("test_minian4", "test")]:
    with open(
        os.path.join(an4_root, "etc", "an4_" + x[1] + ".transcription")
    ) as transcript_f, open(
        os.path.join("data", x[0], "wav.scp"), "w"
    ) as wav_scp_f, open(
        os.path.join("data", x[0], "utt2lang"), "w"
    ) as utt2lang_f:
        wav_scp_f.truncate()
        utt2lang_f.truncate()

        lines = sorted(transcript_f.readlines(), key=lambda s: s.split(" ")[0])
        for line in lines:
            line = line.strip()
            if not line:
                continue
            words = re.search(r"^(.*) \(", line).group(1)
            if words[:4] == "<s> ":
                words = words[4:]
            if words[-5:] == " </s>":
                words = words[:-5]
            source = re.search(r"\((.*)\)", line).group(1)
            pre, mid, last = source.split("-")
            utt_id = "-".join([mid, pre, last])

            lang = mid
            if x[0] == "test_minian4":
                # The langs of test set should be in the train set.
                if mid not in train_langs:
                    # random select a lang from train_langs
                    lang = random.choice(list(train_langs))
                    utt_id = "-".join([lang, utt_id])

            wav_scp_f.write(
                utt_id
                + " "
                + sph2pipe
                + " -f wav -p -c 1 "
                + os.path.join(an4_root, "wav", sph_dir[x[1]], mid, source + ".sph")
                + " |\n"
            )
            # Use speaker id as the language id.
            utt2lang_f.write(utt_id + " " + lang + "\n")
            if x[0] == "train_minian4":
                train_langs.add(lang)
