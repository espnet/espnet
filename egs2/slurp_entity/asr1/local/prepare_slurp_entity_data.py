#!/usr/bin/env python3

# Copyright 2020 University of Stuttgart (Pavel Denisov)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import json
import os
import re
import subprocess
import sys

idir = sys.argv[1]

spk = {}

with open(os.path.join(idir, "dataset", "slurp", "metadata" + ".json")) as meta:
    records = json.load(meta)
    for record in records.values():
        for filename in record["recordings"].keys():
            spk[filename[6:-5]] = record["recordings"][filename]["usrid"]
recordid_unique = {}
for subset in ["train", "devel", "test"]:
    odir = os.path.join("data", subset)
    os.makedirs(odir, exist_ok=True)

    with open(os.path.join(idir, "dataset", "slurp", subset + ".jsonl")) as meta, open(
        os.path.join(odir, "text"), "w", encoding="utf-8"
    ) as text, open(os.path.join(odir, "wav.scp"), "w") as wavscp, open(
        os.path.join(odir, "utt2spk"), "w"
    ) as utt2spk:

        for line in meta:
            prompt = json.loads(line.strip())
            transcript = prompt["sentence"]
            sentence_annotation = prompt["sentence_annotation"]
            num_entities = sentence_annotation.count("[")
            entities = []
            for slot in range(num_entities):
                ent_type = (
                    sentence_annotation.split("[")[slot + 1]
                    .split("]")[0]
                    .split(":")[0]
                    .strip()
                )
                filler = (
                    sentence_annotation.split("[")[slot + 1]
                    .split("]")[0]
                    .split(":")[1]
                    .strip()
                )
                entities.append({"type": ent_type, "filler": filler})
            sortednames = sorted(entities, key=lambda x: x["type"].lower())
            print(sortednames)
            # exit()
            transcript = transcript.replace("@", " at ")
            transcript = transcript.replace("#", " hashtag ")
            transcript = transcript.replace(",", "")
            transcript = transcript.replace(".", "")
            transcript = re.sub(" +", " ", transcript)
            predict_sent = prompt["scenario"] + "_" + prompt["action"]
            for k in sortednames:
                predict_sent += " SEP " + k["type"] + " FILL " + k["filler"].lower()
            predict_sent += " SEP " + transcript
            words = "{}".format(predict_sent).replace("<unk>", "unknown")
            for recording in prompt["recordings"]:
                recoid = recording["file"][6:-5]
                if recoid in recordid_unique:
                    print("Already covered")
                    continue
                recordid_unique[recoid] = 1
                wav = os.path.join(idir, "audio", "slurp_real", recording["file"])
                speaker = spk[recoid]
                uttid = "slurp_{}_{}".format(speaker, recoid)
                text.write("{} {}\n".format(uttid, words))
                utt2spk.write("{} slurp_{}\n".format(uttid, speaker))
                wavscp.write("{} {}\n".format(uttid, wav))
        if subset == "train":
            meta = open(os.path.join(idir, "dataset", "slurp", "train_synthetic.jsonl"))
            for line in meta:
                prompt = json.loads(line.strip())
                transcript = prompt["sentence"]
                sentence_annotation = prompt["sentence_annotation"]
                num_entities = sentence_annotation.count("[")
                entities = []
                for slot in range(num_entities):
                    ent_type = (
                        sentence_annotation.split("[")[slot + 1]
                        .split("]")[0]
                        .split(":")[0]
                        .strip()
                    )
                    filler = (
                        sentence_annotation.split("[")[slot + 1]
                        .split("]")[0]
                        .split(":")[1]
                        .strip()
                    )
                    entities.append({"type": ent_type, "filler": filler})
                sortednames = sorted(entities, key=lambda x: x["type"].lower())
                print(sortednames)
                # exit()
                transcript = transcript.replace("@", " at ")
                transcript = transcript.replace("#", " hashtag ")
                transcript = transcript.replace(",", "")
                transcript = transcript.replace(".", "")
                transcript = re.sub(" +", " ", transcript)
                predict_sent = prompt["scenario"] + "_" + prompt["action"]
                for k in sortednames:
                    predict_sent += " SEP " + k["type"] + " FILL " + k["filler"].lower()
                predict_sent += " SEP " + transcript
                words = "{}".format(predict_sent).replace("<unk>", "unknown")
                for recording in prompt["recordings"]:
                    recoid = recording["file"][6:-5]
                    if recoid in recordid_unique:
                        print("Already covered")
                        continue
                    recordid_unique[recoid] = 1
                    wav = os.path.join(idir, "audio", "slurp_synth", recording["file"])
                    speaker = "synthetic"
                    uttid = "slurp_{}_{}".format(speaker, recoid)
                    text.write("{} {}\n".format(uttid, words))
                    utt2spk.write("{} slurp_{}\n".format(uttid, speaker))
                    wavscp.write("{} {}\n".format(uttid, wav))
