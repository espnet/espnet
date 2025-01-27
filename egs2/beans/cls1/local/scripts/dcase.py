from collections import defaultdict
import sys
from pathlib import Path
import os
import logging
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

from utils import (
    divide_waveform_to_chunks,
    divide_annotation_to_chunks,
    get_wav_length_in_secs,
    BeansRecognitionDataset,
)

DATA_READ_ROOT = sys.argv[1]  # data/dcase
DATA_WRITE_ROOT = sys.argv[2]

CHUNK_SIZE = 60  # in seconds
TARGET_SAMPLE_RATE = 16_000  # for all datasets we fix 16kHz


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

all_wav_paths = sorted(
    Path(os.path.join(DATA_READ_ROOT, "Development_Set")).glob("**/*.wav")
)
for wav_path in tqdm(all_wav_paths, desc="Processing all wav files and annotations."):
    csv_path = wav_path.parent / (wav_path.stem + ".csv")
    logger.info(f"Converting {wav_path} and {csv_path} ...")

    # Break into 60 sec chunks
    target_paths = divide_waveform_to_chunks(
        path=wav_path,
        target_dir=os.path.join(DATA_WRITE_ROOT, "wav"),
        chunk_size=CHUNK_SIZE,
        target_sample_rate=TARGET_SAMPLE_RATE,
    )
    logger.info(f"num_chunks = {len(target_paths)}")

    df = pd.read_csv(csv_path)
    # Add all positive annotations
    annotations = []
    for _, row in df.iterrows():
        st, ed = row["Starttime"], row["Endtime"]

        for species, label in row.iloc[3:].items():
            if label == "POS":
                if species in {"AGGM", "SOCM"}:
                    # these species have very few annotations and will result in
                    # zero samples in either train or test sets after split
                    continue
                annotations.append({"st": st, "ed": ed, "label": species})

    # chunks is defaultdict(list) with keys as chunk index and values as list of annotations
    chunks = divide_annotation_to_chunks(annotations=annotations, chunk_size=CHUNK_SIZE)

    for chunk, path in enumerate(target_paths):
        split = get_split(chunk, len(target_paths))
        # key names are imp, don't change. used in BeansRecognitionDataset
        datasets[split].append(
            {
                "path": path,
                "length": get_wav_length_in_secs(path),
                "annotations": chunks[chunk],
                # empty list if no POS label for this chunk
            }
        )
        # Each item is a wav file of length at most 60 seconds

split2dataset = {
    "dcase.dev": BeansRecognitionDataset(
        dataset=datasets["valid"],
        target_dir=os.path.join(DATA_WRITE_ROOT, "wav_dev"),
        window_width=2,
        window_shift=1,
    ),
    "dcase.train": BeansRecognitionDataset(
        dataset=datasets["train"] + datasets["train-low"],
        target_dir=os.path.join(DATA_WRITE_ROOT, "wav_train"),
        window_width=2,
        window_shift=1,
    ),
    "dcase.test": BeansRecognitionDataset(
        dataset=datasets["test"],
        target_dir=os.path.join(DATA_WRITE_ROOT, "wav_test"),
        window_width=2,
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
            print(f"{uttid} {row['path']}", file=wav_f)
            print(f"{uttid} {' '.join(row['label'])}", file=text_f)
            print(f"{uttid} dummy", file=utt2spk_f)
