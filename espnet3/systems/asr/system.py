# system_asr.py

import logging
import os
from pathlib import Path

from espnet3.systems.asr.decode import decode
from espnet3.systems.asr.tokenizer.utils import (
    gather_training_text,
    train_sentencepiece,
)
from espnet3.systems.base.system import BaseSystem

logger = logging.getLogger(__name__)


class ASRSystem(BaseSystem):
    """ASR-specific system.

    This system adds:
      - Tokenizer training inside train()
    """

    def train(self, collect_stats: bool = False, dataset_dir: str = None):
        logger.info("ASRSystem.train(): starting training process")

        # Train tokenizer if not trained previously
        tokenizer_path = (
            Path(self.train_config.tokenizer.save_path)
            / f"{self.train_config.tokenizer.model_type}.model"
        )
        if not tokenizer_path.exists():
            self._train_tokenizer(dataset_dir=dataset_dir)

        # Proceed with standard training
        return super().train(collect_stats=collect_stats)

    def _train_tokenizer(self, dataset_dir: str = None):
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

    # ----------------------------
    # Evaluate
    # ----------------------------
    def decode(self):
        return decode(self.eval_config)
