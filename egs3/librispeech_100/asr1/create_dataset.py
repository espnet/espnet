# User should create this type of script to prepare dataset,
# and upload to huggingface library.
import os

import datasets
import soundfile as sf
from datasets import Dataset, DatasetDict
from tqdm import tqdm


def parse_transcript_file(transcript_path):
    transcript_dict = {}
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            utt_id, *words = line.strip().split()
            transcript_dict[utt_id] = " ".join(words)
    return transcript_dict


def gather_examples(data_dir):
    examples = {"id": [], "audio": [], "text": []}
    for root, dirs, files in tqdm(os.walk(data_dir)):
        for file in files:
            if file.endswith(".trans.txt"):
                transcript_path = os.path.join(root, file)
                transcript_dict = parse_transcript_file(transcript_path)

                for utt_id, text in transcript_dict.items():
                    flac_path = os.path.join(root, utt_id + ".flac")
                    if not os.path.exists(flac_path):
                        continue
                    examples["id"].append(utt_id)
                    examples["audio"].append({"path": flac_path})
                    examples["text"].append(text)
    return examples


# root path to the dataset（eg：LibriSpeech/train-clean-100）
BASE_PATH = os.environ["LIBRISPEECH_PATH"]

dataset_dict = DatasetDict(
    {
        "train-clean-100": Dataset.from_dict(
            gather_examples(f"{BASE_PATH}/train-clean-100")
        ),
        "dev-clean": Dataset.from_dict(gather_examples(f"{BASE_PATH}/dev-clean")),
        "dev-other": Dataset.from_dict(gather_examples(f"{BASE_PATH}/dev-other")),
        "test-clean": Dataset.from_dict(gather_examples(f"{BASE_PATH}/test-clean")),
        "test-other": Dataset.from_dict(gather_examples(f"{BASE_PATH}/test-other")),
    }
)

# Cast to Audio()
print("Cast to Audio()")
for split in dataset_dict:
    dataset_dict[split] = dataset_dict[split].cast_column(
        "audio", datasets.Audio(decode=False)
    )

dataset_dir = os.environ["LIBRISPEECH"]
print("Saved dataset to", dataset_dir)
dataset_dict.save_to_disk(dataset_dir)
