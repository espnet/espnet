import json
import os
import random
import sys

random.seed(10)

data_path = sys.argv[1]
mixture_base = sys.argv[2]
data_type = sys.argv[3]

raw_scp = os.path.join(data_path, "raw_wav.scp")
wav_scp = os.path.join(data_path, "wav.scp")

# read meta_json and build dic: old speech => new speech
speech_dic = {}
if data_type == "train":
    mix_pathes = [
        os.path.join(mixture_base, "tr_real"),
        os.path.join(mixture_base, "tr_synthetic"),
    ]
elif data_type == "devel":
    mix_pathes = [os.path.join(mixture_base, "cv")]
elif data_type == "test":
    mix_pathes = [os.path.join(mixture_base, "tt")]
elif data_type == "test_qut":
    mix_pathes = [os.path.join(mixture_base, "tt_qut")]

for mix_path in mix_pathes:
    meta_json = os.path.join(mix_path, "metadata.json")
    with open(meta_json, "r") as f:
        data = json.load(f)
        for dic in data:
            old_wav = dic["speech"][0]["source"]["file"].split("/")[-1]
            speech_dic[old_wav] = (mix_path, dic["id"] + ".wav")

# read raw_scp (id, speech)
wav_lines = []
with open(raw_scp, "r") as oldf:
    scps = oldf.readlines()
    # replace speech with new speech
    for line in scps:
        key, old_speech = line.split()
        assert old_speech.split("/")[-1] in speech_dic
        mix_path, new_speech = speech_dic[old_speech.split("/")[-1]]
        if data_type in ["train"]:
            wav_lines.append(
                key
                + " "
                + os.path.join(
                    mix_path, random.choice(["mixture", "s0_dry"]), new_speech
                )
                + "\n"
            )
        elif data_type in ["devel", "test", "test_qut"]:
            wav_lines.append(
                key + " " + os.path.join(mix_path, "mixture", new_speech) + "\n"
            )

with open(wav_scp, "w") as newf2:
    newf2.writelines(wav_lines)
