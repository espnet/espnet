# system_asr.py

import logging
import os
from importlib import import_module
from pathlib import Path
import time
from typing import Iterable

from espnet3.systems.asr.tokenizer.sentencepiece import train_sentencepiece
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
        start = time.perf_counter()
        cfg = getattr(self.train_config, "create_dataset", None)
        if cfg is None or not getattr(cfg, "func", None):
            raise RuntimeError(
                "train_config.create_dataset.func must be set to run create_dataset"
            )
        fn = load_function(cfg.func)
        extra = {k: v for k, v in cfg.items() if k != "func"}
        logger.info("Creating dataset with function %s", cfg.func)
        result = fn(**extra)
        logger.info(
            "Dataset creation completed in %.2fs using %s",
            time.perf_counter() - start,
            cfg.func,
        )
        return result

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
        start = time.perf_counter()
        output_path = Path(self.train_config.tokenizer.save_path)
        output_path.mkdir(parents=True, exist_ok=True)
        tokenizer_cfg = getattr(self.train_config, "tokenizer", None)
        builder_cfg = getattr(tokenizer_cfg, "text_builder", None) if tokenizer_cfg else None
        if builder_cfg is None or not getattr(builder_cfg, "func", None):
            raise RuntimeError(
                "train_config.tokenizer.text_builder.func must be set to build tokenizer text."
            )
        builder = load_function(builder_cfg.func)
        builder_kwargs = {k: v for k, v in builder_cfg.items() if k != "func"}
        logger.info("Building tokenizer training text via %s", builder_cfg.func)
        built = builder(**builder_kwargs)
        texts: list[str]
        if isinstance(built, (str, os.PathLike)):
            path = Path(built)
            if not path.exists():
                raise RuntimeError(f"Tokenizer text file not found: {path}")
            texts = path.read_text(encoding="utf-8").splitlines()
        elif isinstance(built, Iterable):
            texts = [str(t) for t in built]
        else:
            raise RuntimeError(
                f"text_builder must return a path or iterable of strings (got {type(built)})."
            )

        if len(texts) == 0:
            raise RuntimeError(
                "Tokenizer text_builder returned no text. Check dataset preparation."
            )
        logger.info("Collected %d transcript lines for tokenizer training", len(texts))
        with open(output_path / "train.txt", "w", encoding="utf-8") as f:
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
        logger.info("Tokenizer training completed in %.2fs", time.perf_counter() - start)
