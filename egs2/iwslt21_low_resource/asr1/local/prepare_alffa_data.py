#!/usr/bin/env python3

# Copyright 2021 University of Stuttgart (Pavel Denisov)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import sys
from shutil import copyfile

idir = sys.argv[1]

for subset in ["train", "test"]:
    odir = "data/{}_alffa".format(subset)
    os.makedirs(odir, exist_ok=True)

    with open(os.path.join(odir, "text"), "w", encoding="utf-8") as otext, open(
        os.path.join(idir, "data", subset, "text"), encoding="utf-8"
    ) as itext:
        for line in itext:
            line = line.replace("<UNK>", "<spn>")
            line = line.replace(".", " ")
            line = line.replace("?", " ")
            line = line.replace("g20", "g twenty")
            parts = line.strip().split(maxsplit=1)

            if (
                len(parts) == 2 and parts[1] in ["<laughter>", "<music>", "<spn>"]
            ) or parts[1][:3] == "16k":
                continue

            parts[1] = parts[1].replace("-", " ")
            otext.write("{}\n".format(" ".join(parts)))

    with open(os.path.join(odir, "wav.scp"), "w") as owavscp, open(
        os.path.join(idir, "data", subset, "wav.scp")
    ) as iwavscp:
        for line in iwavscp:
            parts = line.strip().split(maxsplit=1)
            parts[1] = parts[1].replace("/my_dir/wav/", "asr_swahili/data/test/wav5/")
            parts[1] = os.path.join(idir, parts[1][12:])
            owavscp.write("{} {}\n".format(parts[0], parts[1]))

    copyfile(
        os.path.join(idir, "data", subset, "utt2spk"), os.path.join(odir, "utt2spk")
    )
