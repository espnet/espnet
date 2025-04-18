from omegaconf import OmegaConf
from hydra.utils import instantiate
from tqdm import tqdm

from espnet3.data import DataOrganizer
from espnet3.preprocess import train_sentencepiece


def load_line(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def main(config_path):
    # Load config
    OmegaConf.register_new_resolver("load_line", load_line)
    config = OmegaConf.load(config_path)
    
    # For training dataset we don't need preprocessor
    config.dataset.preprocessor = None
    organizer = instantiate(config.dataset)

    with open("train_text.txt", "w", encoding="utf-8") as f:
        for example in tqdm(organizer.train):
            f.write(example[1]["text"] + "\n")
            f.flush()
    
    train_sentencepiece(
        dump_text_path="train_text.txt",
        output_path="sentencepiece_model",
        vocab_size=config.vocab_size,
        character_coverage=0.995,
        model_type="bpe",
    )
    

if __name__ == "__main__":
    config_path = "egs3/librispeech_100/asr1/config.yaml"
    main(config_path)
