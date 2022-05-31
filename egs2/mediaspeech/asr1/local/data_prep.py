import argparse
import glob
import json
import math
import os
import os.path
import random

parser = argparse.ArgumentParser(description="Prepare mediaspeech")
parser.add_argument(
    "--data_path", type=str, help="Path to the directory containing all files"
)
parser.add_argument("--train_dir", type=str, help="Path to the train data")
parser.add_argument("--dev_dir", type=str, help="Path to the dev data")
parser.add_argument("--test_dir", type=str, help="Path to the test data")
parser.add_argument("--validated_dir", type=str, help="Path to the validated data")
parser.add_argument("--dev_ratio", type=float, default=0.136, help="Ratio of dev set")
parser.add_argument("--test_ratio", type=float, default=0.136, help="Ratio of test set")
parser.add_argument(
    "--validated_ratio", type=float, default=0.136, help="Ratio of validated set"
)
args = parser.parse_args()

file_names = [
    name[:-5] for name in os.listdir(args.data_path) if name.endswith(".flac")
]
samp_cnt = len(file_names)
dev_samp_cnt = int(samp_cnt * args.dev_ratio)
test_samp_cnt = int(samp_cnt * args.test_ratio)
validated_samp_cnt = int(samp_cnt * args.validated_ratio)
train_samp_cnt = samp_cnt - dev_samp_cnt - test_samp_cnt - validated_samp_cnt

print(
    "samp_cnt, dev_samp_cnt, test_samp_cnt, validated_samp_cnt, train_samp_cnt: ",
    samp_cnt,
    dev_samp_cnt,
    test_samp_cnt,
    validated_samp_cnt,
    train_samp_cnt,
)
print("file_names: ", file_names[:5])

random.seed(2022)
random.shuffle(file_names)
# train_file_names = file_names[:train_samp_cnt]
# dev_file_names = file_names[train_samp_cnt: train_samp_cnt + dev_samp_cnt]
# test_file_names = file_names[train_samp_cnt + dev_samp_cnt:]

train_samples = []
dev_samples = []
test_samples = []
validated_samples = []

for file_idx in range(samp_cnt):
    file_name = file_names[file_idx]
    text_file_path = os.path.join(args.data_path, file_name + ".txt")
    aud_file_path = os.path.join(args.data_path, file_name + ".flac")

    with open(text_file_path, "r") as f:
        text = f.readlines()[0]

    processed_sample = {
        "user_id": 0,
        "text": text,
        "id": file_idx,
        "abs_path": "ffmpeg -i %s -f wav -ar 16000 -ab 16 -ac 1 - |"
        % os.path.abspath(aud_file_path),
    }
    if file_idx < train_samp_cnt:
        train_samples.append(processed_sample)
    elif file_idx < train_samp_cnt + dev_samp_cnt:
        dev_samples.append(processed_sample)
    elif file_idx < train_samp_cnt + dev_samp_cnt + validated_samp_cnt:
        validated_samples.append(processed_sample)
    else:
        test_samples.append(processed_sample)

for setname in ["train", "dev", "test", "validated"]:
    if setname == "train":
        sample_list = train_samples
        dest_dir = args.train_dir
    elif setname == "dev":
        sample_list = dev_samples
        dest_dir = args.dev_dir
    elif setname == "test":
        sample_list = test_samples
        dest_dir = args.test_dir
    elif setname == "validated":
        sample_list = validated_samples
        dest_dir = args.validated_dir
    else:
        raise RuntimeError

    with open(os.path.join(dest_dir, "text"), "w") as text_f, open(
        os.path.join(dest_dir, "wav.scp"), "w"
    ) as wav_scp_f, open(os.path.join(dest_dir, "utt2spk"), "w") as utt2spk_f, open(
        os.path.join(dest_dir, "utt2gender"), "w"
    ) as utt2gndr_f:
        for sample in sample_list:
            text_f.write(f"{int(sample['id']):08d} {sample['text']}\n")
            wav_scp_f.write(f"{int(sample['id']):08d} {sample['abs_path']}\n")
            utt2spk_f.write(f"{int(sample['id']):08d} {int(sample['id']):08d}\n")
            utt2gndr_f.write(f"{int(sample['id']):08d}" + " f\n")
