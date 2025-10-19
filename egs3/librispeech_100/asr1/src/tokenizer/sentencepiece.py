"""Utilities for training SentencePiece tokenizers within the recipe."""

from __future__ import annotations

from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from espnet3.preprocess.sentencepiece import train_sentencepiece


def build_training_corpus(dataset_cfg: DictConfig, output_dir: Path, filename: str = "train_text.txt") -> Path:
    """Dump all transcripts from the training split into a plain-text corpus."""

    cfg = OmegaConf.create(OmegaConf.to_container(dataset_cfg, resolve=True))
    if hasattr(cfg, "preprocessor"):
        cfg.preprocessor = None

    organizer = instantiate(cfg)
    if not hasattr(organizer, "train") or organizer.train is None:
        raise RuntimeError("Tokenizer training requires the training split in the dataset config.")

    text_path = Path(output_dir) / filename
    text_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = organizer.train
    with text_path.open("w", encoding="utf-8") as f:
        for idx in tqdm(range(len(dataset)), desc="Collect transcripts"):
            if hasattr(dataset, "get_text"):
                line = dataset.get_text(idx)
            else:
                sample = dataset[idx]
                line = sample.get("text", "")
            f.write(f"{line}\n")
    return text_path


def train_sentencepiece_from_config(cfg: DictConfig) -> Path:
    """Train SentencePiece model according to ``cfg.tokenizer`` and ``cfg.paths``."""

    tokenizer_dir = Path(cfg.paths.tokenizer_dir)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    dump_path = build_training_corpus(cfg.dataset, tokenizer_dir)

    tokenizer_cfg = cfg.tokenizer
    vocab_size = tokenizer_cfg.get("vocab_size", cfg.get("vocab_size", 5000))
    character_coverage = tokenizer_cfg.get("character_coverage", 1.0)
    model_type = tokenizer_cfg.get("model_type", "bpe")

    train_sentencepiece(
        dump_text_path=dump_path,
        output_path=tokenizer_dir,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
    )

    return tokenizer_dir / f"{model_type}.model"