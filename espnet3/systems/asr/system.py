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

    def create_dataset(self, *args, **kwargs):
        self._reject_stage_args("create_dataset", args, kwargs)
        logger.info("ASRSystem.create_dataset(): starting dataset creation process")
        cfg = getattr(self.train_config, "create_dataset", None)
        if cfg is None or not getattr(cfg, "func", None):
            raise RuntimeError(
                "train_config.create_dataset.func must be set to run create_dataset"
            )
        fn = load_function(cfg.func)
        extra = {k: v for k, v in cfg.items() if k != "func"}
        logger.info(f"Creating dataset with function {cfg.func}")
        return fn(**extra)

    def train(self, *args, **kwargs):
        self._reject_stage_args("train", args, kwargs)
        logger.info("ASRSystem.train(): starting training process")

        dataset_dir = getattr(self.train_config, "dataset_dir", None)
        if dataset_dir is None:
            raise RuntimeError("train_config.dataset_dir must be set for training.")

        # Train tokenizer if not trained previously
        tokenizer_path = (
            Path(self.train_config.tokenizer.save_path)
            / f"{self.train_config.tokenizer.model_type}.model"
        )
        if not tokenizer_path.exists():
            self.train_tokenizer()

        # Proceed with standard training
        return super().train()

    def train_tokenizer(self, *args, **kwargs):
        self._reject_stage_args("train_tokenizer", args, kwargs)
        dataset_dir = getattr(self.train_config, "dataset_dir", None)
        if dataset_dir is None:
            raise RuntimeError("train_config.dataset_dir must be set for tokenizer training.")

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
