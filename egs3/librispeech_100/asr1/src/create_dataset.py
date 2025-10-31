import argparse
import os
from pathlib import Path

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
    examples = {"id": [], "speech": [], "text": []}
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
                    examples["speech"].append({"path": flac_path})
                    examples["text"].append(text)
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="Path to the raw LibriSpeech-like dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="data/",
        help="Directory where the HuggingFace dataset will be saved",
    )
    args = parser.parse_args()

    splits = ["train-clean-100", "dev-clean", "dev-other", "test-clean", "test-other"]

    dataset_dict = DatasetDict()
    for split in splits:
        split_path = os.path.join(args.dataset_dir, split)
        print(f"Gathering examples from: {split_path}")
        dataset = Dataset.from_dict(gather_examples(split_path))
        # Uncomment below if you want to cast to datasets.Audio
        # dataset = dataset.cast_column("audio", datasets.Audio(decode=False))
        dataset_dict[split] = dataset

    print("Saving dataset to", args.output_dir)
    dataset_dict.save_to_disk(args.output_dir)


if __name__ == "__main__":
    main()
