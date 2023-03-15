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
    asr_transcript_file = open(
        "/ocean/projects/cis210027p/siddhana/espnet/egs2/slurp_new/asr1/"
        + "exp/asr_train_asr_conformer_raw_en_word/"
        + "inference_asr_model_valid.acc.ave_10best/"
        + subset
        + "/score_wer/hyp.trn",
        "r",
    )
    transcript_arr = [line for line in asr_transcript_file]
    transcript_dict = {}
    for line in transcript_arr:
        if "synth" in line:
            wav_name = "-".join(line.split("\t")[1].strip()[1:-1].split("-")[1:])
        else:
            wav_name = "-".join(line.split("\t")[1].strip()[1:-1].split("-")[2:])
        sub_word_transcript = " ".join(line.split("\t")[0].split()[1:]).lower()
        if sub_word_transcript == "":
            sub_word_transcript = "blank"
        text = sub_word_transcript.split()[0].replace("▁", "")
        for sub_word in sub_word_transcript.split()[1:]:
            if "▁" in sub_word:
                text = text + " " + sub_word.replace("▁", "")
            else:
                text = text + sub_word
        transcript_dict[wav_name] = text
    with open(os.path.join(idir, "dataset", "slurp", subset + ".jsonl")) as meta, open(
        os.path.join(odir, "text"), "w", encoding="utf-8"
    ) as text, open(
        os.path.join(odir, "transcript"), "w", encoding="utf-8"
    ) as transcript_file, open(
        os.path.join(odir, "wav.scp"), "w"
    ) as wavscp, open(
        os.path.join(odir, "utt2spk"), "w"
    ) as utt2spk:
        for line in meta:
            prompt = json.loads(line.strip())
            transcript = prompt["sentence"]
            transcript = transcript.replace("@", " at ")
            transcript = transcript.replace("#", " hashtag ")
            transcript = transcript.replace(",", "")
            transcript = transcript.replace(".", "")
            transcript = re.sub(" +", " ", transcript).replace("<unk>", "unknown")
            words = "{}".format(
                prompt["scenario"] + "_" + prompt["action"] + " " + transcript
            ).replace("<unk>", "unknown")
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
                transcript_file.write("{} {}\n".format(uttid, transcript_dict[uttid]))
                utt2spk.write("{} slurp_{}\n".format(uttid, speaker))
                wavscp.write("{} {}\n".format(uttid, wav))
        if subset == "train":
            meta = open(os.path.join(idir, "dataset", "slurp", "train_synthetic.jsonl"))
            for line in meta:
                prompt = json.loads(line.strip())
                transcript = prompt["sentence"]
                transcript = transcript.replace("@", " at ")
                transcript = transcript.replace("#", " hashtag ")
                transcript = transcript.replace(",", "")
                transcript = transcript.replace(".", "")
                transcript = (
                    re.sub(" +", " ", transcript).lower().replace("<unk>", "unknown")
                )
                words = "{}".format(
                    prompt["scenario"] + "_" + prompt["action"] + " " + transcript
                ).replace("<unk>", "unknown")
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
                    transcript_file.write(
                        "{} {}\n".format(uttid, transcript_dict[uttid])
                    )
                    utt2spk.write("{} slurp_{}\n".format(uttid, speaker))
                    wavscp.write("{} {}\n".format(uttid, wav))
