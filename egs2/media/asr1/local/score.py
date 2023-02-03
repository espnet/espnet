#!/usr/bin/env python3

# Copyright 2022, University of Stuttgart (Pavel Denisov)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import sys

import jiwer

filenames = {"ref": "", "hyp": ""}
filenames["ref"] = sys.argv[1]
filenames["hyp"] = sys.argv[2]

slots = {"concept": {"ref": {}, "hyp": {}}, "concept_value": {"ref": {}, "hyp": {}}}

for f in ["ref", "hyp"]:
    with open(filenames[f], encoding="utf-8") as file:
        for line in file:
            parts = line.split("\t")

            if " SEP " in parts[0]:
                sem = [
                    x.strip().replace(" ", "_") for x in parts[0].split(" SEP ")[:-1]
                ]
                concept = " ".join([x.split("_FILL_", maxsplit=1)[0] for x in sem])
                concept_value = " ".join(sem)
            else:
                concept = ""
                concept_value = ""

            slots["concept"][f][parts[1]] = concept
            slots["concept_value"][f][parts[1]] = concept_value

slots_tuples = {"concept": [], "concept_value": []}

for utt in slots["concept"]["ref"].keys():
    if slots["concept"]["ref"][utt] != "":
        slots_tuples["concept"].append(
            (slots["concept"]["ref"][utt], slots["concept"]["hyp"][utt])
        )
        slots_tuples["concept_value"].append(
            (slots["concept_value"]["ref"][utt], slots["concept_value"]["hyp"][utt])
        )

cer = (
    jiwer.wer(
        [x[0] for x in slots_tuples["concept"]], [x[1] for x in slots_tuples["concept"]]
    )
    * 100.0
)
cver = (
    jiwer.wer(
        [x[0] for x in slots_tuples["concept_value"]],
        [x[1] for x in slots_tuples["concept_value"]],
    )
    * 100.0
)

print(f"CER;CVER: {cer:.2f};{cver:.2f}")
