import argparse
import os
from pathlib import Path

from tqdm import tqdm

from espnet3.asr.tokenizer.sentencepiece import train_sentencepiece


def parse_transcript_file(transcript_path):
    transcript_dict = {}
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            utt_id, *words = line.strip().split()
            transcript_dict[utt_id] = " ".join(words)
    return transcript_dict


def gather_training_text(data_dir):
    texts = []
    for root, dirs, files in tqdm(os.walk(data_dir)):
        for file in files:
            if file.endswith(".trans.txt"):
                transcript_path = os.path.join(root, file)
                transcript_dict = parse_transcript_file(transcript_path)

                for utt_id, text in transcript_dict.items():
                    texts.append(text)

    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="Path to the raw LibriSpeech-like dataset directory",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        default="data/tokenizer",
        help="Directory to save the tokenizers",
    )
    parser.add_argument(
        "--vocab_size", type=int, default=5000, help="The size of vocab"
    )
    parser.add_argument(
        "--model_type", type=str, default="bpe", help="The type of the tokenizer"
    )
    args = parser.parse_args()

    split_path = os.path.join(args.dataset_dir, "train-clean-100")
    output_path = Path(args.save_path)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Gathering examples from: {split_path}")
    texts = gather_training_text(split_path)
    with open(output_path / "train.txt", "w") as f:
        f.write("\n".join(texts))

    train_sentencepiece(
        args.save_path / "train.txt",
        args.save_path,
        args.vocab_size,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()
