#!/usr/bin/env python3

import json
import os
import re
import subprocess
import sys

slurp_dir = sys.argv[1]
slurp_s_dir = sys.argv[2]
libritrans_s_dir = sys.argv[3]

spk = {}
# Prepare slurp speaker
with open(os.path.join(slurp_dir, "dataset", "slurp", "metadata" + ".json")) as meta:
    records = json.load(meta)
    for record in records.values():
        for filename in record["recordings"].keys():
            spk[filename[6:-5]] = record["recordings"][filename]["usrid"]

libritrans_dirs = {
    "train": ["tr"],
    "devel": ["cv"],
    "lt_test": ["tt"],
    "lt_test_qut": ["tt_qut"],
}

slurp_dirs = {
    "train": ["tr_real", "tr_synthetic"],
    "devel": ["cv"],
    "slurp_test": ["tt"],
    "slurp_test_qut": ["tt_qut"],
}

recordid_unique = {}
for subset in [
    "train",
    "devel",
    "lt_test",
    "lt_test_qut",
    "slurp_test",
    "slurp_test_qut",
]:
    odir = os.path.join("data", subset)
    os.makedirs(odir, exist_ok=True)

    with open(os.path.join(odir, "text"), "w", encoding="utf-8") as text, open(
        os.path.join(odir, "spk1.scp"), "w"
    ) as spkscp, open(os.path.join(odir, "wav.scp"), "w") as wavscp, open(
        os.path.join(odir, "utt2spk"), "w"
    ) as utt2spk:
        # preparing LibriTrans-S data
        if subset in libritrans_dirs:
            for subdir in libritrans_dirs[subset]:
                mix_path = os.path.join(libritrans_s_dir, subdir)
                meta_json = os.path.join(mix_path, "metadata.json")
                with open(meta_json, "r") as f:
                    data = json.load(f)
                    for dic in data:
                        uttid = (
                            "libritrans_"
                            + dic["speech"][0]["source"]["file"].split("/")[-1][:-4]
                        )
                        speaker = dic["speech"][0]["source"]["spkid"]

                        # writing
                        utt2spk.write("{} libritrans_{}\n".format(uttid, speaker))
                        text.write("{} {}\n".format(uttid, "dummy"))
                        spkscp.write(
                            "{} {}\n".format(
                                uttid,
                                os.path.join(mix_path, "s0_dry", dic["id"] + ".wav"),
                            )
                        )
                        wavscp.write(
                            "{} {}\n".format(
                                uttid,
                                os.path.join(mix_path, "mixture", dic["id"] + ".wav"),
                            )
                        )

        # preparing SLURP-S data
        if subset in slurp_dirs:
            for subdir in slurp_dirs[subset]:
                mix_path = os.path.join(slurp_s_dir, subdir)
                meta_json = os.path.join(mix_path, "metadata.json")
                with open(meta_json, "r") as f:
                    data = json.load(f)
                    for dic in data:
                        utt_name = dic["speech"][0]["source"]["file"].split("/")[-1]
                        recoid = utt_name[6:-5]
                        # skipped covered speech
                        if recoid in recordid_unique:
                            continue
                        elif subset in ["train", "devel"]:
                            recordid_unique[recoid] = 1
                        if subdir == "tr_synthetic":
                            speaker = "synthetic"
                        else:
                            speaker = spk[recoid]

                        # writing
                        uttid = "slurp_{}_{}".format(speaker, recoid)
                        utt2spk.write("{} slurp_{}\n".format(uttid, speaker))
                        text.write("{} {}\n".format(uttid, "dummy"))
                        spkscp.write(
                            "{} {}\n".format(
                                uttid,
                                os.path.join(mix_path, "s0_dry", dic["id"] + ".wav"),
                            )
                        )
                        wavscp.write(
                            "{} {}\n".format(
                                uttid,
                                os.path.join(mix_path, "mixture", dic["id"] + ".wav"),
                            )
                        )
