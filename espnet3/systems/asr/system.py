"""ASR system implementation and tokenizer training helpers.

This module adds ASR-specific stages on top of the base system, primarily
tokenizer training support.
"""

import logging
import os
import time
from importlib import import_module
from pathlib import Path
from typing import Iterable

from espnet3.systems.asr.tokenizers.sentencepiece import train_sentencepiece
from espnet3.systems.base.system import BaseSystem

logger = logging.getLogger(__name__)


class ASRSystem(BaseSystem):
    """ASR-specific system.

    This system adds:
      - Tokenizer training inside train()

    Additional stage log paths:
        | Stage           | Path reference                  |
        |---              |---                              |
        | train_tokenizer | training_config.tokenizer.save_path |
    """

    def __init__(
        self,
        training_config=None,
        inference_config=None,
        metrics_config=None,
        **kwargs,
    ) -> None:
        """Initialize the ASR system with ASR-specific stage mappings."""
        super().__init__(
            training_config=training_config,
            inference_config=inference_config,
            metrics_config=metrics_config,
            stage_log_mapping={
                "train_tokenizer": "training_config.tokenizer.save_path",
            },
            **kwargs,
        )

    def train(self, *args, **kwargs):
        """Train the model, training the tokenizer first if needed.

        This stage checks for a cached tokenizer model and runs tokenizer
        training before delegating to the base training routine.

        Raises:
            RuntimeError: If neither dataset references nor ``dataset_dir`` exist.
        """
        self._reject_stage_args("train", args, kwargs)
        logger.info("ASRSystem.train(): starting training process")

        dataset_dir = getattr(self.training_config, "dataset_dir", None)
        dataset_config = getattr(self.training_config, "dataset", None)
        if dataset_dir is None and dataset_config is None:
            raise RuntimeError(
                "training_config.dataset or training_config.dataset_dir must be set "
                "for training."
            )

        # Train tokenizer if not trained previously
        if not self._has_tokenizer():
            self.train_tokenizer()

        # Proceed with standard training
        return super().train()

    def _has_tokenizer(self) -> bool:
        tokenizer_config = self.training_config.tokenizer
        output_path = Path(tokenizer_config.save_path)
        model = output_path / f"{tokenizer_config.model_type}.model"
        vocab = output_path / f"{tokenizer_config.model_type}.vocab"
        return model.exists() and vocab.exists()

    def train_tokenizer(self, *args, **kwargs):
        """Train a SentencePiece tokenizer based on configured text.

        The text builder configured in ``training_config.tokenizer.text_builder``
        is used to generate training text, which is then saved and consumed
        by the SentencePiece trainer.

        Raises:
            RuntimeError: If required tokenizer config is missing or invalid.
        """
        self._reject_stage_args("train_tokenizer", args, kwargs)

        if self._has_tokenizer():
            logger.info("Tokenizer already exists. Skipping train_tokenizer().")
            return
        start = time.perf_counter()
        tokenizer_config = getattr(self.training_config, "tokenizer", None)
        builder_config = (
            getattr(tokenizer_config, "text_builder", None)
            if tokenizer_config
            else None
        )
        if builder_config is None or not getattr(builder_config, "func", None):
            raise RuntimeError(
                "training_config.tokenizer.text_builder.func must be set to build "
                "tokenizer text."
            )
        module_path, func_name = builder_config.func.rsplit(".", 1)
        builder = getattr(import_module(module_path), func_name)
        builder_kwargs = {k: v for k, v in builder_config.items() if k != "func"}
        logger.info("Building tokenizer training text via %s", builder_config.func)
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
                "text_builder must return a path or iterable of strings "
                f"(got {type(built)})."
            )

        if len(texts) == 0:
            raise RuntimeError(
                "Tokenizer text_builder returned no text. Check dataset preparation."
            )
        output_path = Path(self.training_config.tokenizer.save_path)
        output_path.mkdir(parents=True, exist_ok=True)
        train_text_path = getattr(tokenizer_config, "train_file", None)
        if train_text_path:
            train_text_path = Path(train_text_path)
        else:
            train_text_path = output_path / "train.txt"
        if train_text_path.exists():
            raise RuntimeError(
                f"Tokenizer training text already exists: {train_text_path}"
            )
        train_text_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Collected %d transcript lines for tokenizer training", len(texts))
        with open(train_text_path, "w", encoding="utf-8") as f:
            f.write("\n".join(texts))

        logger.info(f"Training tokenizer: {self.training_config.tokenizer.model_type}")
        logger.info(f"Tokenizer output: {self.training_config.tokenizer.save_path}")

        # Example placeholder:
        train_sentencepiece(
            train_text_path,
            output_path,
            self.training_config.tokenizer.vocab_size,
            model_type=self.training_config.tokenizer.model_type,
        )
        logger.info(
            "Tokenizer training completed in %.2fs", time.perf_counter() - start
        )
