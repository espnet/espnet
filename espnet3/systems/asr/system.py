# system_asr.py

import logging
import os
from importlib import import_module
from pathlib import Path

from espnet3.systems.asr.tokenizer.utils import (
    gather_training_text,
    train_sentencepiece,
)
from espnet3.systems.base.system import BaseSystem

logger = logging.getLogger(__name__)


def load_function(path):
    module_path, func_name = path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, func_name)


class ASRSystem(BaseSystem):
    """ASR-specific system.

    This system adds:
      - Tokenizer training inside train()
    """

    def create_dataset(self, func: str, **kwargs):
        logger.info("ASRSystem.create_dataset(): starting dataset creation process")
        logger.info(f"Creating dataset with function {func}")

        # We assume that dataset preparation script should be implemented in src/create_dataset.py
        fn = load_function(func)
        return fn(**kwargs)

    def train(self, dataset_dir: str = None):
        logger.info("ASRSystem.train(): starting training process")

        # Train tokenizer if not trained previously
        tokenizer_path = (
            Path(self.train_config.tokenizer.save_path)
            / f"{self.train_config.tokenizer.model_type}.model"
        )
        if not tokenizer_path.exists():
            self.train_tokenizer(dataset_dir=dataset_dir)

        # Proceed with standard training
        return super().train()

    def train_tokenizer(self, dataset_dir: str = None):
        assert (
            dataset_dir is not None
        ), "dataset_dir must be provided for tokenizer training"

        split_path = os.path.join(dataset_dir, "train-clean-100")
        output_path = Path(self.train_config.tokenizer.save_path)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Gathering examples from: {split_path}")

        texts = gather_training_text(split_path)
        with open(output_path / "train.txt", "w") as f:
            f.write("\n".join(texts))

        logger.info(f"Training tokenizer: {self.train_config.tokenizer.model_type}")
        logger.info(f"Tokenizer output: {self.train_config.tokenizer.save_path}")

        # Example placeholder:
        train_sentencepiece(
            output_path / "train.txt",
            output_path,
            self.train_config.tokenizer.vocab_size,
            model_type=self.train_config.tokenizer.model_type,
        )
        logger.info("Tokenizer training completed.")
