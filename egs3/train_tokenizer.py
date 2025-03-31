# User should create this type of script to prepare tokenizer.
from tqdm import tqdm
from datasets import load_from_disk
import datasets
from espnet3.preprocess import train_sentencepiece

if __name__ == "__main__":
    dataset_dir = "/home/msomeki/workspace/librispeech_dataset"
    dataset = load_from_disk(
        dataset_dir,
    )['train-clean-100']

    dataset = dataset.cast_column("audio", datasets.Audio(decode=False))

    with open("train_text.txt", "w", encoding="utf-8") as f:
        for example in tqdm(dataset):
            f.write(example["text"] + "\n")
            f.flush()

    train_sentencepiece(
        dump_text_path="train_text.txt",
        output_path="sentencepiece_model",
        vocab_size=6500,
        character_coverage=0.995,
        model_type="bpe",
    )

