#!/usr/bin/env python3

# Copyright 2020 University of Stuttgart (Pavel Denisov)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import json
import os
import sys
import subprocess
import re

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
    ) as utt2spk, open(os.path.join(odir,"text.intent"), "w") as intfile, open(os.path.join(odir,"text.ner"), "w") as entfile:

        for line in meta:
            prompt = json.loads(line.strip())
            sent = prompt["sentence"]
            if "<unk>" in sent:
                continue
            #transcript = transcript.replace(",", "")
            #transcript = transcript.replace(".", "")
            words = []
            for token in prompt["tokens"]:
                words.append(token["surface"])
            #transcript = re.sub(" +", " ", transcript)
            transcript = " ".join(words)
            intent = "{}".format(prompt["scenario"] + "_" + prompt["action"])
            #words = transcript.split(" ")
            entities = ["Other"] * len(words)
            for entity in prompt["entities"]:
                span = entity["span"]
                ent = entity["type"]
                entities[span[0]]="B_"+ent
                for sp in span[1:]:
                    entities[sp]="I_"+ent
            entities = " ".join(entities)
            for recording in prompt["recordings"]:
                recoid = recording["file"][6:-5]
                if recoid in recordid_unique:
                    print("Already covered")
                    continue
                recordid_unique[recoid] = 1
                wav = os.path.join(idir, "audio", "slurp_real", recording["file"])
                speaker = spk[recoid]
                uttid = "slurp_{}_{}".format(speaker, recoid)
                text.write("{} {}\n".format(uttid, transcript))
                entfile.write("{} {}\n".format(uttid, entities))
                intfile.write("{} {}\n".format(uttid, intent))
                utt2spk.write("{} slurp_{}\n".format(uttid, speaker))
                wavscp.write("{} {}\n".format(uttid, wav))
        if subset == "train":
            meta = open(os.path.join(idir, "dataset", "slurp", "train_synthetic.jsonl"))
            for line in meta:
                prompt = json.loads(line.strip())
                sent = prompt["sentence"]
                if "<unk>" in sent:
                    continue
                #transcript = transcript.replace(",", "")
                #transcript = transcript.replace(".", "")
                words = []
                for token in prompt["tokens"]:
                    words.append(token["surface"])
                #transcript = re.sub(" +", " ", transcript)
                transcript = " ".join(words)
                intent = "{}".format(prompt["scenario"] + "_" + prompt["action"])
                #words = transcript.split(" ")
                entities = ["Other"] * len(words)
                for entity in prompt["entities"]:
                    span = entity["span"]
                    ent = entity["type"]
                    entities[span[0]]="B_"+ent
                    for sp in span[1:]:
                        entities[sp]="I_"+ent
                entities = " ".join(entities)
                for recording in prompt["recordings"]:
                    recoid = recording["file"][6:-5]
                    if recoid in recordid_unique:
                        print("Already covered")
                        continue
                    recordid_unique[recoid] = 1
                    wav = os.path.join(idir, "audio", "slurp_synth", recording["file"])
                    speaker = "synthetic"
                    uttid = "slurp_{}_{}".format(speaker, recoid)
                    text.write("{} {}\n".format(uttid, transcript))
                    entfile.write("{} {}\n".format(uttid, entities))
                    intfile.write("{} {}\n".format(uttid, intent))
                    utt2spk.write("{} slurp_{}\n".format(uttid, speaker))
                    wavscp.write("{} {}\n".format(uttid, wav))
