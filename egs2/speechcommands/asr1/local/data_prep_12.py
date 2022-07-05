#!/usr/bin/env python3

# Copyright 2021 Carnegie Mellon University (Yifan Peng)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Speech Commands Dataset: https://arxiv.org/abs/1804.03209
# Our data preparation is similar to the TensorFlow script:
# https://www.tensorflow.org/datasets/catalog/speech_commands


import argparse
import csv
import glob
import os
import os.path

import numpy as np
from scipy.io import wavfile

parser = argparse.ArgumentParser(description="Process speech commands dataset.")
parser.add_argument(
    "--data_path",
    type=str,
    default="downloads/speech_commands_v0.02",
    help="folder containing the original data",
)
parser.add_argument(
    "--test_data_path",
    type=str,
    default="downloads/speech_commands_test_set_v0.02",
    help="folder containing the test set",
)
parser.add_argument(
    "--train_dir",
    type=str,
    default="data/train",
    help="output folder for training data",
)
parser.add_argument(
    "--dev_dir", type=str, default="data/dev", help="output folder for validation data"
)
parser.add_argument(
    "--test_dir", type=str, default="data/test", help="output folder for test data"
)
parser.add_argument(
    "--speechbrain_testcsv",
    type=str,
    default="local/speechbrain_test.csv",
    help="speechbrain test csv file",
)
parser.add_argument(
    "--speechbrain_test_dir",
    type=str,
    default="data/test_speechbrain",
    help="output folder for speechbrain test data",
)
args = parser.parse_args()


WORDS = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]
SILENCE = "_silence_"
UNKNOWN = "_unknown_"
LABELS = WORDS + [SILENCE, UNKNOWN]  # 12 labels in the test set
BACKGROUND_NOISE = "_background_noise_"
SAMPLE_RATE = 16000

# Generate test data
with open(os.path.join(args.test_dir, "text"), "w") as text_f, open(
    os.path.join(args.test_dir, "wav.scp"), "w"
) as wav_scp_f, open(os.path.join(args.test_dir, "utt2spk"), "w") as utt2spk_f:
    for label in LABELS:
        wav_list = [
            n
            for n in os.listdir(os.path.join(args.test_data_path, label))
            if n.endswith(".wav")
        ]
        for wav in wav_list:
            uttid = f'{label.strip("_")}_{wav.rstrip(".wav")}'
            text_f.write(uttid + " " + label + "\n")
            wav_scp_f.write(
                uttid
                + " "
                + os.path.abspath(os.path.join(args.test_data_path, label, wav))
                + "\n"
            )
            utt2spk_f.write(uttid + " " + uttid + "\n")

# Generate train and dev data
with open(os.path.join(args.data_path, "validation_list.txt"), "r") as dev_f:
    dev_file_list = [line.rstrip() for line in dev_f.readlines()]
    # add running_tap into the dev set
    dev_file_list.append(os.path.join(BACKGROUND_NOISE, "running_tap.wav"))
    dev_file_list = [
        os.path.abspath(os.path.join(args.data_path, line)) for line in dev_file_list
    ]
with open(os.path.join(args.data_path, "testing_list.txt"), "r") as test_f:
    test_file_list = [line.rstrip() for line in test_f.readlines()]
    test_file_list = [
        os.path.abspath(os.path.join(args.data_path, line)) for line in test_file_list
    ]

full_file_list = [
    os.path.abspath(p) for p in glob.glob(os.path.join(args.data_path, "*", "*.wav"))
]
train_file_list = list(set(full_file_list) - set(dev_file_list) - set(test_file_list))


# UNKOWN is around 18 times as large as any other word
def filter_file_list(file_list, ratio=18, excluded=WORDS + [BACKGROUND_NOISE]):
    file_dict = {}
    for p in file_list:
        w = p.split("/")[-2]  # word, i.e. folder name
        if w not in file_dict:
            file_dict[w] = []
        file_dict[w].append(p)

    new_file_list = []
    for w in file_dict:
        if w in excluded:
            new_file_list += file_dict[w]
        else:
            new_file_list += sorted(file_dict[w])[::ratio]  # every `ratio` files
    return new_file_list


for name in ["train", "dev"]:
    if name == "train":
        file_list = train_file_list
        out_dir = args.train_dir
    else:
        file_list = dev_file_list
        out_dir = args.dev_dir

    # filter the list to reduce unknown words
    file_list = filter_file_list(file_list)

    with open(os.path.join(out_dir, "text"), "w") as text_f, open(
        os.path.join(out_dir, "wav.scp"), "w"
    ) as wav_scp_f, open(os.path.join(out_dir, "utt2spk"), "w") as utt2spk_f:
        for wav_abspath in file_list:  # absolute path
            word, wav = wav_abspath.split("/")[-2:]
            if word != BACKGROUND_NOISE:
                if word in WORDS:
                    label = word
                else:
                    label = UNKNOWN
                uttid = f'{word.strip("_")}_{wav.rstrip(".wav")}'
                text_f.write(uttid + " " + label + "\n")
                wav_scp_f.write(uttid + " " + wav_abspath + "\n")
                utt2spk_f.write(uttid + " " + uttid + "\n")
            else:
                processed_dir = os.path.join(
                    args.data_path, BACKGROUND_NOISE, "processed"
                )
                os.makedirs(processed_dir, exist_ok=True)
                label = SILENCE

                # split the original audio to 1-second clips
                wav_rate, wav_data = wavfile.read(wav_abspath)  # 1-D array
                assert wav_rate == SAMPLE_RATE
                for start in range(
                    0, wav_data.shape[0] - SAMPLE_RATE, SAMPLE_RATE // 9
                ):
                    audio_segment = wav_data[start : start + SAMPLE_RATE]
                    uttid = f'{wav.rstrip(".wav")}_{start:08d}'
                    wavfile.write(
                        os.path.join(processed_dir, f"{uttid}.wav"),
                        SAMPLE_RATE,
                        audio_segment,
                    )
                    text_f.write(uttid + " " + label + "\n")
                    wav_scp_f.write(
                        uttid
                        + " "
                        + os.path.abspath(os.path.join(processed_dir, f"{uttid}.wav"))
                        + "\n"
                    )
                    utt2spk_f.write(uttid + " " + uttid + "\n")

# Generate SpeechBrain test data
with open(args.speechbrain_testcsv, "r") as f:
    speechbrain_lines = list(csv.reader(f))[1:]  # remove header line

with open(os.path.join(args.speechbrain_test_dir, "text"), "w") as text_f, open(
    os.path.join(args.speechbrain_test_dir, "wav.scp"), "w"
) as wav_scp_f, open(
    os.path.join(args.speechbrain_test_dir, "utt2spk"), "w"
) as utt2spk_f:
    for sb_line in speechbrain_lines:
        sb_id, _, start, stop, sb_wav = sb_line[:5]
        command = sb_line[10]
        wav_path = os.path.join(args.data_path, "/".join(sb_wav.split("/")[-2:]))
        if command == "silence":
            speechbrain_processed_dir = os.path.join(
                os.path.split(wav_path)[0], "speechbrain_processed"
            )
            os.makedirs(speechbrain_processed_dir, exist_ok=True)
            # extract audio segment
            wav_rate, wav_data = wavfile.read(wav_path)
            assert wav_rate == SAMPLE_RATE
            audio_segment = wav_data[int(start) : int(stop)]

            uttid = "_".join(["silence"] + sb_id.split("/")[1:])
            wav_save_path = os.path.abspath(
                os.path.join(speechbrain_processed_dir, uttid + ".wav")
            )
            wavfile.write(wav_save_path, SAMPLE_RATE, audio_segment)

            text_f.write(uttid + " " + "_silence_" + "\n")
            wav_scp_f.write(uttid + " " + wav_save_path + "\n")
            utt2spk_f.write(uttid + " " + uttid + "\n")
        else:
            wav_save_path = os.path.abspath(
                os.path.join(args.data_path, "/".join(sb_wav.split("/")[-2:]))
            )
            if command == "unknown":
                uttid = "_".join(["unknown"] + sb_id.split("/"))
                command = "_unknown_"
            else:
                uttid = "_".join(sb_id.split("/"))

            text_f.write(uttid + " " + command + "\n")
            wav_scp_f.write(uttid + " " + wav_save_path + "\n")
            utt2spk_f.write(uttid + " " + uttid + "\n")
