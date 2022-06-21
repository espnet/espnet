#!/usr/bin/env python3

# Copyright 2020 University of Stuttgart (Pavel Denisov)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import json
import os
import sys
import subprocess
import re
from jiwer import wer

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
        os.path.join(odir, "text.asr.en"), "w", encoding="utf-8"
    ) as asr_text, open(
        os.path.join(odir, "text.ner.en"), "w", encoding="utf-8"
    ) as ner_text, open(os.path.join(odir, "wav.scp"), "w") as wavscp, open(
        os.path.join(odir, "utt2spk"), "w"
    ) as utt2spk:

        for line in meta:
            prompt = json.loads(line.strip())
            transcript = prompt["sentence"]
            sentence_annotation = prompt["sentence_annotation"]
            num_entities = sentence_annotation.count("[")
            transcript = re.sub(" +", " ", transcript)
            sentence_annotation = re.sub(" +", " ", sentence_annotation)
            entities = []
            transcript_arr=[]
            for slot in range(num_entities):
                if slot==0:
                    word_arr = sentence_annotation.split("[")[slot].strip().split(" ")
                else:
                    word_arr = sentence_annotation.split("[")[slot].split("]")[1].strip().split(" ")
                for word in word_arr:
                    if word=="":
                        continue
                    transcript_arr.append(word)
                    entities.append("na")
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
                entities.append(ent_type+"_B")
                transcript_arr.append(filler.strip().split(" ")[0])
                for word in filler.strip().split(" ")[1:]:
                    transcript_arr.append(word)
                    entities.append(ent_type+"_I")
            for word in sentence_annotation.split("]")[-1].strip().split(" "):
                if word=="":
                    continue
                transcript_arr.append(word)
                entities.append("na")
            assert len(entities)== len(transcript_arr)
            predict_sent = " ".join(entities)
            ner_words = "{}".format(predict_sent).replace("<unk>", "unknown")
            asr_words = "{}".format(" ".join(transcript_arr).lower()).replace("<unk>", "unknown")
            assert len(ner_words.split())== len(asr_words.split())
            for recording in prompt["recordings"]:
                recoid = recording["file"][6:-5]
                if recoid in recordid_unique:
                    print("Already covered")
                    continue
                recordid_unique[recoid] = 1
                wav = os.path.join(idir, "audio", "slurp_real", recording["file"])
                speaker = spk[recoid]
                uttid = "slurp_{}_{}".format(speaker, recoid)
                bad_utts=["slurp_FE-146_1490194174","slurp_FE-146_1490194174-headset","slurp_FE-146_1490194812","slurp_FE-146_1490194812-headset","slurp_FE-348_1490203037","slurp_FE-348_1490203037-headset","slurp_FO-125_1488985276","slurp_FO-219_1434531106-headset","slurp_FO-234_1497884435","slurp_ME-151_1490704565","slurp_ME-151_1490704565-headset","slurp_ME-223_1434541274","slurp_ME-223_1434541274-headset","slurp_ME-223_1434542857","slurp_MO-030_1488539350","slurp_MO-038_1488552957","slurp_MO-038_1488558226","slurp_MO-044_1488562241","slurp_MO-050_1488627933","slurp_MO-051_1488635051","slurp_MO-055_1488637518","slurp_MO-055_1488637959","slurp_MO-142_1490105636","slurp_MO-142_1490105636-headset","slurp_MO-142_1497192558-headset","slurp_MO-422_1500388042","slurp_MO-422_1500388042-headset","slurp_MO-446_1501696074","slurp_UNK-343_1490106848","slurp_UNK-343_1490106848-headset"]
                if uttid in bad_utts:
                    print("remove")
                    continue
                asr_text.write("{} {}\n".format(uttid, asr_words))
                ner_text.write("{} {}\n".format(uttid, ner_words))
                utt2spk.write("{} slurp_{}\n".format(uttid, speaker))
                wavscp.write("{} {}\n".format(uttid, wav))
        if subset == "train":
            meta = open(os.path.join(idir, "dataset", "slurp", "train_synthetic.jsonl"))
            for line in meta:
                prompt = json.loads(line.strip())
                transcript = prompt["sentence"]
                sentence_annotation = prompt["sentence_annotation"]
                num_entities = sentence_annotation.count("[")
                transcript = re.sub(" +", " ", transcript)
                sentence_annotation = re.sub(" +", " ", sentence_annotation)
                entities = []
                transcript_arr=[]
                for slot in range(num_entities):
                    if slot==0:
                        word_arr = sentence_annotation.split("[")[slot].strip().split(" ")
                    else:
                        word_arr = sentence_annotation.split("[")[slot].split("]")[1].strip().split(" ")
                    for word in word_arr:
                        if word=="":
                            continue
                        transcript_arr.append(word)
                        entities.append("na")
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
                    entities.append(ent_type+"_B")
                    transcript_arr.append(filler.strip().split(" ")[0])
                    for word in filler.strip().split(" ")[1:]:
                        transcript_arr.append(word)
                        entities.append(ent_type+"_I")
                for word in sentence_annotation.split("]")[-1].strip().split(" "):
                    if word=="":
                        continue
                    transcript_arr.append(word)
                    entities.append("na")
                # sortednames = sorted(entities, key=lambda x: x["type"].lower())
                assert len(entities)== len(transcript_arr)
                predict_sent = " ".join(entities)
                ner_words = "{}".format(predict_sent).replace("<unk>", "unknown")
                asr_words = "{}".format(" ".join(transcript_arr).lower()).replace("<unk>", "unknown")
                for recording in prompt["recordings"]:
                    recoid = recording["file"][6:-5]
                    if recoid in recordid_unique:
                        print("Already covered")
                        continue
                    recordid_unique[recoid] = 1
                    wav = os.path.join(idir, "audio", "slurp_synth", recording["file"])
                    speaker = "synthetic"
                    uttid = "slurp_{}_{}".format(speaker, recoid)
                    bad_utts=["slurp_synthetic_1590161498769ltcBA-synth",\
                    "slurp_synthetic_1590161499298olocO-synth",\
                    "slurp_synthetic_1590161499837aOCcb-synth",\
                    "slurp_synthetic_1590163373563CBcOt-synth",\
                    "slurp_synthetic_1590163374092bcBlt-synth",\
                    "slurp_synthetic_1590163374636BlbtA-synth",\
                    "slurp_synthetic_1590163375165clCOA-synth",\
                    "slurp_synthetic_1590170435438lAaCb-synth",\
                    "slurp_synthetic_1590170436037lCooO-synth",\
                    "slurp_synthetic_1590168705932tcooO-synth",\
                    "slurp_synthetic_1590168706865bAlOa-synth",\
                    "slurp_synthetic_1590168707670CbcOl-synth",\
                    "slurp_synthetic_1590168708515OtboA-synth",\
                    "slurp_synthetic_1590170436629tbooc-synth"]
                    if uttid in bad_utts:
                        print("remove")
                        print(uttid)
                        continue
                    asr_text.write("{} {}\n".format(uttid, asr_words))
                    ner_text.write("{} {}\n".format(uttid, ner_words))
                    utt2spk.write("{} slurp_{}\n".format(uttid, speaker))
                    wavscp.write("{} {}\n".format(uttid, wav))