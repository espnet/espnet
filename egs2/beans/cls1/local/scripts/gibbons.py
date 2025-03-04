import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from utils import (
    BeansRecognitionDataset,
    divide_annotation_to_chunks,
    divide_waveform_to_chunks,
    get_wav_length_in_secs,
)

logger = logging.getLogger(__name__)

DATA_READ_ROOT = sys.argv[1]  # data/dcase
DATA_WRITE_ROOT = sys.argv[2]

CHUNK_SIZE = 60  # in seconds
TARGET_SAMPLE_RATE = 16000  # for all datasets we fix 16kHz
WINDOW_WIDTH = 4


def get_split(file_id):
    # 1:5:3:3
    if file_id == 0:
        return "train-low"
    elif 1 <= file_id <= 5:
        return "train"
    elif 6 <= file_id <= 8:
        return "valid"
    else:
        return "test"


datasets = defaultdict(list)
os.makedirs(Path(DATA_READ_ROOT).parent / "processed", exist_ok=True)
for file_id, wav_path in enumerate(sorted(Path(DATA_READ_ROOT).glob("*.wav"))):
    print(f"Converting {wav_path} ...", file=sys.stderr)

    target_paths = divide_waveform_to_chunks(
        path=wav_path,
        target_dir=Path(DATA_READ_ROOT).parent / "processed",
        chunk_size=CHUNK_SIZE,
        target_sample_rate=TARGET_SAMPLE_RATE,
    )
    df = pd.read_csv(
        os.path.join(DATA_READ_ROOT, "Train_Labels", ("g_" + wav_path.stem + ".data"))
    )
    annotations = []
    for _, row in df.iterrows():
        st, ed, type = row["Start"], row["End"], row["Type"]
        annotations.append({"st": st, "ed": ed, "label": str(type)})

    chunks = divide_annotation_to_chunks(annotations=annotations, chunk_size=CHUNK_SIZE)
    split = get_split(file_id)

    for chunk, path in enumerate(target_paths):
        if chunk % 3 != 0:
            continue
        datasets[split].append(
            {
                "path": path,
                "length": get_wav_length_in_secs(path),
                "annotations": chunks[chunk],
            }
        )


split2dataset = {
    "gibbons.dev": BeansRecognitionDataset(
        dataset=datasets["valid"],
        target_dir=os.path.join(DATA_WRITE_ROOT, "wav_dev"),
        window_width=WINDOW_WIDTH,
        window_shift=1,
    ),
    "gibbons.train": BeansRecognitionDataset(
        dataset=datasets["train"] + datasets["train-low"],
        target_dir=os.path.join(DATA_WRITE_ROOT, "wav_train"),
        window_width=WINDOW_WIDTH,
        window_shift=1,
    ),
    "gibbons.test": BeansRecognitionDataset(
        dataset=datasets["test"],
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
