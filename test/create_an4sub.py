#!/usr/bin/env python
# coding: utf-8
import json
import os
import shutil

cwd = os.path.dirname(os.path.realpath(__file__)) + "/"
print("working at " + cwd)
dumpdir = cwd + "../egs/an4/asr1/dump/train_dev/deltafalse/"
with open(dumpdir + "data.json") as f:
    utts = json.load(f)["utts"]

ark1 = {k: v for k, v in utts.items() if ".1.ark" in v["input"][0]["feat"]}

for k, v in ark1.items():
    v["input"][0]["feat"] = "test/an4sub.ark:" + \
        v["input"][0]["feat"].split(":")[-1]

json.dump({"utts": ark1}, open(cwd + "an4sub.json", "w"), indent=2)
shutil.copy(dumpdir + "feats.1.ark", cwd + "an4sub.ark")
shutil.copy(cwd + "../egs/an4/asr1/data/lang_1char/train_nodev_units.txt",
            cwd + "an4sub_dict.txt")
