import glob

from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import os
import sys
import logging
import json
from tqdm import tqdm
from utils import (
    BeansRecognitionDataset,
    get_wav_length_in_secs,
)
import pathlib
import subprocess

WINDOW_WIDTH = 10
CHUNK_SIZE = 60 # in seconds
TARGET_SAMPLE_RATE = 16000

logger = logging.getLogger(__name__)

DATA_READ_ROOT = sys.argv[1]  # data/dcase
DATA_WRITE_ROOT = sys.argv[2]

datasets = {}

for split in ["train", "train-low", "valid", "test"]:
    json_data = []
    with open(os.path.join(DATA_READ_ROOT, split + ".jsonl"), "r") as f:
        for line in f:
            if isinstance(line, str):
                json_obj = json.loads(line.strip())
                json_obj["path"] = os.path.join(DATA_READ_ROOT, "wav", Path(json_obj["path"]).stem + ".wav")
                json_data.append(json_obj)

    datasets[split] = json_data


split2dataset = {
    "hiceas.dev": BeansRecognitionDataset(
        dataset=datasets["valid"],
        target_dir=os.path.join(DATA_WRITE_ROOT, "wav_dev"),
        window_width=WINDOW_WIDTH,
        window_shift=1,
    ),
    "hiceas.train": BeansRecognitionDataset(
        dataset=datasets["train"] + datasets["train-low"],
        target_dir=os.path.join(DATA_WRITE_ROOT, "wav_train"),
        window_width=WINDOW_WIDTH,
        window_shift=1,
    ),
    "hiceas.test": BeansRecognitionDataset(
        dataset=datasets["test"],
        target_dir=os.path.join(DATA_WRITE_ROOT, "wav_test"),
        window_width=WINDOW_WIDTH,
        window_shift=1,
    ),
}

for split, dataset in split2dataset.items():
    text_path = os.path.join(DATA_WRITE_ROOT, split, "text")
    wav_path = os.path.join(DATA_WRITE_ROOT, split, "wav.scp")
    utt2spk_path = os.path.join(DATA_WRITE_ROOT, split, "utt2spk")
    os.makedirs(os.path.dirname(text_path), exist_ok=True)
    with open(text_path, "w") as text_f, open(wav_path, "w") as wav_f, open(
        utt2spk_path, "w"
    ) as utt2spk_f:
        dlen = len(dataset)
        for i in tqdm(range(dlen), desc=f"Writing wav.scp and text files for {split}"):
            row = dataset[i]
            uttid = f"{split}-{i}"
            label_seq = " ".join(row["label"])
            if len(label_seq) == 0:
                label_seq = "<blank>"
            print(f"{uttid} {row['path']}", file=wav_f)
            print(f"{uttid} {label_seq}", file=text_f)
            print(f"{uttid} dummy", file=utt2spk_f)