"""ASR system implementation and tokenizer training helpers.

This module adds ASR-specific stages on top of the base system, including
tokenizer training and dataset creation hooks.
"""

import logging
import os
import time
from importlib import import_module
from pathlib import Path
from typing import Iterable

from espnet3.systems.asr.tokenizer.sentencepiece import train_sentencepiece
from espnet3.systems.base.system import BaseSystem

logger = logging.getLogger(__name__)


def load_function(path):
    """Load a callable from a dotted module path.

    Args:
        path: Dotted module path (e.g., ``package.module:function``).

    Returns:
        Callable referenced by the path.

    Raises:
        (Exception): Propagated import or attribute lookup errors.
    """
    module_path, func_name = path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, func_name)


class ASRSystem(BaseSystem):
    """ASR-specific system.

    This system adds:
      - Tokenizer training inside train()
    """

    def create_dataset(self, *args, **kwargs):
        """Create datasets using the configured helper function.

        The callable is resolved from ``train_config.create_dataset.func`` and
        invoked with the remaining configuration values.

        Raises:
            RuntimeError: If the configuration does not specify a function.
        """
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
        """Train the model, training the tokenizer first if needed.

        This stage checks for a cached tokenizer model and runs tokenizer
        training before delegating to the base training routine.

        Raises:
            RuntimeError: If ``train_config.dataset_dir`` is not set.
        """
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
    
    def _tokenizer_exists(self) -> bool:
        tokenizer_cfg = self.train_config.tokenizer
        output_path = Path(tokenizer_cfg.save_path)
        model = output_path / f"{tokenizer_cfg.model_type}.model"
        vocab = output_path / f"{tokenizer_cfg.model_type}.vocab"
        return model.exists() and vocab.exists()

    def train_tokenizer(self, *args, **kwargs):
        """Train a SentencePiece tokenizer based on configured text.

        The text builder configured in ``train_config.tokenizer.text_builder``
        is used to generate training text, which is then saved and consumed
        by the SentencePiece trainer.

        Raises:
            RuntimeError: If required tokenizer config is missing or invalid.
        """
        self._reject_stage_args("train_tokenizer", args, kwargs)

        if self._tokenizer_exists():
            logger.info("Tokenizer already exists. Skipping train_tokenizer().")
            return
    
        start = time.perf_counter()
        output_path = Path(self.train_config.tokenizer.save_path)
        output_path.mkdir(parents=True, exist_ok=True)
        tokenizer_cfg = getattr(self.train_config, "tokenizer", None)
        builder_cfg = (
            getattr(tokenizer_cfg, "text_builder", None) if tokenizer_cfg else None
        )
        if builder_cfg is None or not getattr(builder_cfg, "func", None):
            raise RuntimeError(
                "train_config.tokenizer.text_builder.func must be set to build "
                "tokenizer text."
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
                "text_builder must return a path or iterable of strings "
                f"(got {type(built)})."
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
        logger.info(
            "Tokenizer training completed in %.2fs", time.perf_counter() - start
        )
