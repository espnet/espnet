"""ASR system implementation and tokenizer training helpers.

This module adds ASR-specific stages on top of the base system, primarily
tokenizer training support.
"""

import logging
import os
import time
from importlib import import_module
from pathlib import Path
from typing import ClassVar, Iterable

from espnet3.contract.stages import StageSpec
from espnet3.systems.base.system import BaseSystem
from espnet3.systems.esp2_asr.src.tokenizers.sentencepiece import train_sentencepiece

logger = logging.getLogger(__name__)


class ASRSystem(BaseSystem):
    """ASR-specific system.

    This system adds ``train_tokenizer`` (its log goes under
    ``tokenizer.save_path`` of its own stage config), run before ``train``
    trains the tokenizer if it is not already cached.

    Examples:
        >>> [s.name for s in ASRSystem.stages[:2]]
        ['create_dataset', 'train_tokenizer']
    """

    stages: ClassVar[tuple[StageSpec, ...]] = (
        StageSpec(name="create_dataset", config="training", log_dir="data_dir"),
        StageSpec(
            name="train_tokenizer", config="training", log_dir="tokenizer.save_path"
        ),
        StageSpec(name="collect_stats", config="training", log_dir="stats_dir"),
        StageSpec(name="train", config="training", log_dir="exp_dir"),
        StageSpec(name="infer", config="inference", log_dir="inference_dir"),
        StageSpec(name="measure", config="metrics", log_dir="inference_dir"),
        StageSpec(name="pack_model", config=("training", "publication")),
        StageSpec(name="upload_model", config="publication"),
        StageSpec(name="pack_demo", config="demo", log_dir="pack.out_dir"),
        StageSpec(name="upload_demo", config="demo", log_dir="pack.out_dir"),
    )

    def train(self, *args, **kwargs):
        """Train the model, training the tokenizer first if needed.

        This stage checks for a cached tokenizer model and runs tokenizer
        training before delegating to the base training routine.

        Raises:
            RuntimeError: If neither dataset references nor ``dataset_dir`` exist.

        Examples:
            >>> system = ASRSystem(configs={"training": training_config})
            >>> system.train()
        """
        self._reject_stage_args("train", args, kwargs)
        logger.info("ASRSystem.train(): starting training process")

        config = self.stage_configs["train"]
        dataset_dir = getattr(config, "dataset_dir", None)
        dataset_config = getattr(config, "dataset", None)
        if dataset_dir is None and dataset_config is None:
            raise RuntimeError(
                "train: the stage config has no dataset or dataset_dir; set "
                "one in training.yaml."
            )

        # Train tokenizer if not trained previously
        if not self._has_tokenizer():
            self.train_tokenizer()

        # Proceed with standard training
        return super().train()

    def _has_tokenizer(self) -> bool:
        tokenizer_config = self.stage_configs["train_tokenizer"].tokenizer
        output_path = Path(tokenizer_config.save_path)
        model = output_path / f"{tokenizer_config.model_type}.model"
        vocab = output_path / f"{tokenizer_config.model_type}.vocab"
        return model.exists() and vocab.exists()

    def train_tokenizer(self, *args, **kwargs):
        """Train a SentencePiece tokenizer based on configured text.

        The text builder configured in this stage's own
        ``tokenizer.text_builder`` is used to generate training text, which
        is then saved and consumed by the SentencePiece trainer.

        Raises:
            RuntimeError: If required tokenizer config is missing or invalid.

        Examples:
            >>> system = ASRSystem(configs={"training": training_config})
            >>> system.train_tokenizer()
        """
        self._reject_stage_args("train_tokenizer", args, kwargs)

        if self._has_tokenizer():
            logger.info("Tokenizer already exists. Skipping train_tokenizer().")
            return
        start = time.perf_counter()
        config = self.stage_configs["train_tokenizer"]
        tokenizer_config = getattr(config, "tokenizer", None)
        builder_config = (
            getattr(tokenizer_config, "text_builder", None)
            if tokenizer_config
            else None
        )
        if builder_config is None or not getattr(builder_config, "func", None):
            raise RuntimeError(
                "train_tokenizer: the stage config has no "
                "tokenizer.text_builder.func; set it in training.yaml."
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
        output_path = Path(tokenizer_config.save_path)
        output_path.mkdir(parents=True, exist_ok=True)
        train_text_path = getattr(tokenizer_config, "train_file", None)
        if train_text_path:
            train_text_path = Path(train_text_path)
        else:
            data_dir = getattr(config, "data_dir", None)
            if data_dir:
                train_text_path = Path(data_dir) / "train_tokenizer" / "train.txt"
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

        logger.info("Training tokenizer: %s", tokenizer_config.model_type)
        logger.info("Tokenizer output: %s", tokenizer_config.save_path)

        # Example placeholder:
        train_sentencepiece(
            train_text_path,
            output_path,
            tokenizer_config.vocab_size,
            model_type=tokenizer_config.model_type,
        )
        logger.info(
            "Tokenizer training completed in %.2fs", time.perf_counter() - start
        )
