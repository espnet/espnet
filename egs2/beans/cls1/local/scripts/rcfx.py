from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import os
import sys
import logging
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

dataset = {}
os.makedirs(Path(DATA_READ_ROOT).parent / "processed", exist_ok=True)
dest_dir = pathlib.Path(os.path.join(Path(DATA_READ_ROOT).parent, "processed"))
for src_file in sorted(pathlib.Path(os.path.join(DATA_READ_ROOT, 'train')).glob('*.flac')):
    dest_file = dest_dir / (src_file.stem + '.wav')
    if not os.path.exists(dest_file):
        print(f"Converting {src_file} ...", file=sys.stderr)
        subprocess.run(
            ['sox', src_file, f'-r {TARGET_SAMPLE_RATE}', '-R', dest_file]
        )
    dataset[src_file.stem] = {
        'path': str(dest_file),
        'length': get_wav_length_in_secs(dest_file),
        'annotations': []}


df = pd.read_csv(os.path.join(DATA_READ_ROOT, 'train_tp.csv'))
for _, row in df.iterrows():
    dataset[row['recording_id']]['annotations'].append(
        {'st': row['t_min'], 'ed': row['t_max'], 'label': str(row['species_id'])})

# split to train:valid:test = 6:2:2
dataset = list(dataset.values())
df_train, df_valid_test = train_test_split(dataset, test_size=0.4, random_state=42, shuffle=True)
df_valid, df_test = train_test_split(df_valid_test, test_size=0.5, random_state=42, shuffle=True)
df_train_low, _ = train_test_split(df_train, test_size=0.66, random_state=42, shuffle=True)

df_train.sort(key=lambda x: x['path'])
df_train_low.sort(key=lambda x: x['path'])
df_valid.sort(key=lambda x: x['path'])
df_test.sort(key=lambda x: x['path'])


split2dataset = {
    "rcfx.dev": BeansRecognitionDataset(
        dataset=df_valid,
        target_dir=os.path.join(DATA_WRITE_ROOT, "wav_dev"),
        window_width=WINDOW_WIDTH,
        window_shift=1,
    ),
    "rcfx.train": BeansRecognitionDataset(
        dataset=df_train + df_train_low,
        target_dir=os.path.join(DATA_WRITE_ROOT, "wav_train"),
        window_width=WINDOW_WIDTH,
        window_shift=1,
    ),
    "rcfx.test": BeansRecognitionDataset(
        dataset=df_test,
        target_dir=os.path.join(DATA_WRITE_ROOT, "wav_test"),
        window_width=WINDOW_WIDTH,
        window_shift=1,
    ),
}

# This can be copied once split2dataset is a dict of BeansRecognitionDataset
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


