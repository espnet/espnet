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

WINDOW_WIDTH = 2
CHUNK_SIZE = 60  # in seconds
TARGET_SAMPLE_RATE = 16000

logger = logging.getLogger(__name__)

DATA_READ_ROOT = sys.argv[1]  # data/dcase
DATA_WRITE_ROOT = sys.argv[2]


def get_split(chunk_id, total_num_chunks):
    if chunk_id / total_num_chunks < 0.12:
        return "train-low"
    elif chunk_id / total_num_chunks < 0.6:
        return "train"
    elif chunk_id / total_num_chunks < 0.8:
        return "valid"
    else:
        return "test"


datasets = defaultdict(list)
os.makedirs(Path(DATA_READ_ROOT).parent / "processed", exist_ok=True)
for wav_path in sorted(Path(DATA_READ_ROOT).glob("Recording_?/*.wav")):
    print(f"Converting {wav_path} ...", file=sys.stderr)

    target_paths = divide_waveform_to_chunks(
        path=wav_path,
        target_dir=Path(DATA_READ_ROOT).parent / "processed",
        chunk_size=CHUNK_SIZE,
        target_sample_rate=TARGET_SAMPLE_RATE,
    )

    df = pd.read_csv(
        os.path.join(
            DATA_READ_ROOT,
            "_".join(wav_path.stem.split("_")[:2]),
            wav_path.stem + ".Table.1.selections.txt",
        ),
        sep="\t",
    )

    annotations = []
    for _, row in df.iterrows():
        st, ed = row["Begin Time (s)"], row["End Time (s)"]
        annotations.append({"st": st, "ed": ed, "label": str(row["Species"])})

    chunks = divide_annotation_to_chunks(annotations=annotations, chunk_size=CHUNK_SIZE)

    for chunk, path in enumerate(target_paths):
        split = get_split(chunk, len(target_paths))
        datasets[split].append(
            {
                "path": path,
                "length": get_wav_length_in_secs(path),
                "annotations": chunks[chunk],
            }
        )


split2dataset = {
    "enabirds.dev": BeansRecognitionDataset(
        dataset=datasets["valid"],
        target_dir=os.path.join(DATA_WRITE_ROOT, "wav_dev"),
        window_width=WINDOW_WIDTH,
        window_shift=1,
    ),
    "enabirds.train": BeansRecognitionDataset(
        dataset=datasets["train"] + datasets["train-low"],
        target_dir=os.path.join(DATA_WRITE_ROOT, "wav_train"),
        window_width=WINDOW_WIDTH,
        window_shift=1,
    ),
    "enabirds.test": BeansRecognitionDataset(
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
