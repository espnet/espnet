#!/usr/bin/env python3

# Copyright 2020 Carnegie Mellon University (Siddhant Arora)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import json
import os
import sys
import subprocess
import re

for subset in ["train", "devel", "test"]:
    odir = os.path.join("data", subset)
    with open( os.path.join(odir, "text.asr.en"), "r", encoding="utf-8"
    ) as asr_text, open( os.path.join(odir, "text.ner.en"), "r", encoding="utf-8"
    ) as ner_text, open( os.path.join(odir, "text_subtoken.ner.en"), "w", encoding="utf-8"
    ) as ner_subtoken_text:
        asr_line_arr = [line for line in asr_text]
        ner_line_arr = [line for line in ner_text]
        for asr_line_count in range(len(asr_line_arr)):
            asr_line = asr_line_arr[asr_line_count].strip()
            ner_line = ner_line_arr[asr_line_count].strip().split(" ")[1:]
            ner_subtoken_line=[ner_line_arr[asr_line_count].strip().split(" ")[0]]
            count=0
            for word in asr_line.split(" ")[1:]:
                if word[0] == "▁":
                    # print(word)
                    # print(count)
                    # print(asr_line)
                    # print(ner_line)
                    # print(ner_line[count])
                    ner_subtoken_line.append(ner_line[count])
                    count += 1
                else:
                    ner_subtoken_line.append("FILL")
            assert len(ner_subtoken_line) == len(asr_line.split(" "))
            ner_subtoken_text.write(" ".join(ner_subtoken_line)+"\n")
            
