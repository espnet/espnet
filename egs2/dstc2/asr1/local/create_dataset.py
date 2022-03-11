import os
import glob
import json
import pandas as pd
import shutil

###### DOWNLOAD DATASET
import os

audio_urls = {
    "audio_Apr11_S0": "https://github.com/matthen/dstc/releases/download/v1/audio_Apr11_S0.tar.gz",
    "audio_Apr11_S1": "https://github.com/matthen/dstc/releases/download/v1/audio_Apr11_S1.tar.gz",
    "audio_Apr11_S2": "https://github.com/matthen/dstc/releases/download/v1/audio_Apr11_S2.tar.gz",
    "audio_Apr11_S3": "https://github.com/matthen/dstc/releases/download/v1/audio_Apr11_S3.tar.gz",
    "audio_Mar13_S0A0": "https://github.com/matthen/dstc/releases/download/v1/audio_Mar13_S0A0.tar.gz",
    "audio_Mar13_S0A1": "https://github.com/matthen/dstc/releases/download/v1/audio_Mar13_S0A1.tar.gz",
    "audio_Mar13_S1A0": "https://github.com/matthen/dstc/releases/download/v1/audio_Mar13_S1A0.tar.gz",
    "audio_Mar13_S1A1": "https://github.com/matthen/dstc/releases/download/v1/audio_Mar13_S1A1.tar.gz",
    "audio_Mar13_S2A0": "https://github.com/matthen/dstc/releases/download/v1/audio_Mar13_S2A0.tar.gz",
    "audio_Mar13_S2A1": "https://github.com/matthen/dstc/releases/download/v1/audio_Mar13_S2A1.tar.gz",
}

if not os.path.isdir("audios"):
    os.mkdir("audios")

if not os.path.isdir("json_files"):
    os.mkdir("json_files")

if not os.path.isdir("files_list"):
    os.mkdir("files_list")


os.chdir("audios")

for dir in audio_urls.keys():
    if not os.path.isdir(dir):
        os.mkdir(dir)
    os.chdir(dir)
    os.system("wget " + audio_urls[dir])
    os.system("tar -xvzf *.tar.gz")
    os.chdir("..")


os.chdir("..")

data_urls = {
    "dstc2_traindev.tar.gz": "https://github.com/matthen/dstc/releases/download/v1/dstc2_traindev.tar.gz",
    "dstc2_test.tar.gz": "https://github.com/matthen/dstc/releases/download/v1/dstc2_test.tar.gz",
}

for key in data_urls.keys():
    os.system("wget " + data_urls[key])
    while not os.path.exists(key):
        time.sleep(10)

    os.system("tar -xvzf " + key)
    os.system("mv  data/* json_files")
    os.system("mv scripts/config/*  files_list")
    os.system("rm -rf data scripts")


##### END OF DOWNLOAD


def get_label(sem):
    """get label from semantic structure"""
    slots = sem["slots"]
    act = sem["act"]

    assert len(slots) in [0, 1]
    if len(slots) == 0:
        label = act
    else:
        slot = slots[0]
        assert len(slot) in [1, 2]
        if len(slot) == 1:
            label = "%s-%s" % (act, slot[0])
        else:
            if act == "request":
                label = "%s-%s" % (act, slot[1])
            else:
                label = "%s-%s-%s" % (act, slot[0], slot[1])

    return label


#
def create_dialog_acts(label_turn):
    json_list = label_turn["semantics"]["json"]
    dialog_acts = []
    for sem in json_list:
        dialog_acts.append(get_label(sem))
    return dialog_acts


def get_n_best_list(log_turn):
    hyp_list = log_turn["input"]["live"]["asr-hyps"]
    return hyp_list


def get_data_fnlist(scp_fn):
    """get file name list"""
    with open(scp_fn, "r") as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines]
    return lines


train_scp_fn = os.path.join("files_list/dstc2_train.flist")
valid_scp_fn = os.path.join("files_list/dstc2_dev.flist")
test_scp_fn = os.path.join("files_list/dstc2_test.flist")


fnlist = {}
fnlist["train"] = get_data_fnlist(train_scp_fn)
fnlist["valid"] = get_data_fnlist(valid_scp_fn)
fnlist["test"] = get_data_fnlist(test_scp_fn)

output_file = {}
output_file["train"] = "train.csv"
output_file["valid"] = "validation.csv"
output_file["test"] = "test.csv"

# create wav directory
if not os.path.exists("wavs/speakers/BrX8aDqK2cLZRYl/"):
    os.makedirs("wavs/speakers/BrX8aDqK2cLZRYl/")

# create data directory
if not os.path.exists("data"):
    os.mkdir("data")


for mode in ["train", "valid", "test"]:
    dataset = []
    visited = set()
    for fn in fnlist[mode]:
        fn = os.path.join("json_files", fn)
        dir_name = fn.split("/")[-1]
        label_fn = os.path.join(fn, "label.json")
        log_fn = os.path.join(fn, "log.json")
        with open(os.path.join(label_fn)) as f:
            data = json.loads(f.read())
            label_turns = data["turns"]
        with open(os.path.join(log_fn)) as f:
            data = json.loads(f.read())
            log_turns = data["turns"]

        for turn_label, turn_log in zip(label_turns, log_turns):
            dialog_acts = create_dialog_acts(turn_label)
            prev_sys_response = turn_log["output"]["transcript"]
            paths = log_fn.split(os.sep)
            audio_path = (
                "audios/"
                + "audio_"
                + paths[1]
                + "/data/"
                + paths[1]
                + "/"
                + dir_name
                + "/"
                + "original_dir/"
                + turn_label["audio-file"]
            )
            shutil.copyfile(
                "audios/"
                + "audio_"
                + paths[1]
                + "/data/"
                + paths[1]
                + "/"
                + dir_name
                + "/"
                + "original_dir/"
                + turn_label["audio-file"],
                "wavs/speakers/BrX8aDqK2cLZRYl/"
                + dir_name
                + "_"
                + turn_label["audio-file"],
            )
            audio_file_path = (
                "wavs/speakers/BrX8aDqK2cLZRYl/"
                + dir_name
                + "_"
                + turn_label["audio-file"]
            )
            asr_hyp = get_n_best_list(turn_log)
            transcript = turn_label["transcription"]
            if dialog_acts:
                dataset.append(
                    [
                        prev_sys_response,
                        transcript,
                        asr_hyp,
                        audio_file_path,
                        dialog_acts,
                    ]
                )

    export_df = pd.DataFrame(
        dataset,
        columns=[
            "prev_sys_utterance",
            "transcript",
            "asr_hypothesis",
            "audio_file_path",
            "dialog_acts",
        ],
    )
    duplicated_df = export_df.loc[export_df.duplicated(subset="audio_file_path"), :]
    duplicated_df.to_csv("duplicates_" + mode + ".csv")
    export_df = export_df.drop_duplicates(subset="audio_file_path")
    export_df.to_csv(os.path.join("data", output_file[mode]))
