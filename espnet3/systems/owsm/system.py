"""OWSM system: one shared vocabulary with the OWSM special symbols reserved."""

import logging
import os
import time
from pathlib import Path
from typing import Iterable, List

from hydra.utils import instantiate
from omegaconf import DictConfig

from espnet3.systems.base.system import BaseSystem
from espnet3.systems.owsm.tokenizers.sentencepiece import train_sentencepiece

logger = logging.getLogger(__name__)


class OWSMSystem(BaseSystem):
    """System for OWSM-style speech-to-text.

    ``espnet2.tasks.s2t.S2TTask`` trains on one vocabulary, like ASR, but its
    text stream is tagged: ``"<eng><asr><0.00> ...<4.74>"``. Those tags have to
    reach SentencePiece as ``user_defined_symbols`` or it splits them into
    ``<``, ``e``, ``n``, ``g``, ``>`` and spends vocabulary on the brackets --
    which is what this subclass adds over a plain ASR tokenizer stage.

    Config shape under ``tokenizer``::

        tokenizer:
          vocab_size: 50000
          model_type: bpe
          save_path: ${data_dir}/bpe_50000
          text_builder: {_target_: ..., ...}

    ``collect_stats`` is extended to make the token shape files 2-D -- see
    :meth:`collect_stats`.
    """

    def __init__(
        self,
        training_config: DictConfig | None = None,
        inference_config: DictConfig | None = None,
        metrics_config: DictConfig | None = None,
        publication_config: DictConfig | None = None,
        stage_log_mapping: dict | None = None,
        demo_config: DictConfig | None = None,
    ) -> None:
        """Initialize the OWSM system with optional stage configs.

        Args:
            training_config: Training configuration.
            inference_config: Inference configuration.
            metrics_config: Measurement configuration.
            publication_config: Publication configuration for the model
                packing and upload stages.
            stage_log_mapping: Optional per-stage log directory overrides.
            demo_config: Demo configuration for the demo stages.
        """
        super().__init__(
            training_config=training_config,
            inference_config=inference_config,
            metrics_config=metrics_config,
            publication_config=publication_config,
            stage_log_mapping={
                "train_tokenizer": "training_config.tokenizer.save_path",
                **(stage_log_mapping or {}),
            },
            demo_config=demo_config,
        )

    def train(self, *args, **kwargs):
        """Train the model, training the tokenizer first when it is missing.

        Raises:
            RuntimeError: If neither ``dataset`` nor ``dataset_dir`` is set on
                the training config.
        """
        self._reject_stage_args("train", args, kwargs)

        dataset_dir = getattr(self.training_config, "dataset_dir", None)
        dataset_config = getattr(self.training_config, "dataset", None)
        if dataset_dir is None and dataset_config is None:
            raise RuntimeError(
                "training_config.dataset or training_config.dataset_dir must be "
                "set for training."
            )

        if not self._has_tokenizer():
            self.train_tokenizer()

        return super().train()

    def collect_stats(self, *args, **kwargs):
        """Collect stats, then make the token shape files 2-D.

        With a large vocabulary the decoder logits dominate GPU memory, so the
        batcher has to weigh a token stream by its length times the vocabulary
        size rather than by length alone. This appends ``V`` to the shape of
        every stream named in ``tokenizer.vocab_scaled_shapes``, which defaults
        to the streams that reach the decoder: ``text`` and ``text_prev``.

        ``text_ctc`` is left 1-D. CTC projects the encoder, so its logits are
        already paid for by the speech shape.
        """
        self._reject_stage_args("collect_stats", args, kwargs)
        result = super().collect_stats()
        self._append_vocab_size_to_text_shapes()
        return result

    def _vocab_size(self) -> int:
        """Vocabulary size: the number of lines in the token list."""
        tokens = Path(self.training_config.tokenizer.save_path) / "tokens.txt"
        with tokens.open(encoding="utf-8") as stream:
            return sum(1 for _ in stream)

    def _append_vocab_size_to_text_shapes(self) -> None:
        """Rewrite each ``<stream>_shape`` from ``L`` to ``L,V`` in place."""
        stats_dir = Path(self.training_config.stats_dir)
        streams = getattr(
            self.training_config.tokenizer,
            "vocab_scaled_shapes",
            ["text", "text_prev"],
        )
        vocab = self._vocab_size()
        for mode in ("train", "valid"):
            mode_dir = stats_dir / mode
            missing = [
                stream
                for stream in streams
                if not (mode_dir / f"{stream}_shape").is_file()
            ]
            if missing:
                raise RuntimeError(
                    f"tokenizer.vocab_scaled_shapes lists {', '.join(missing)}, "
                    f"but no matching shape file exists under {mode_dir}. "
                    "Training reads these shapes to bound the batch size."
                )
            for stream in streams:
                path = mode_dir / f"{stream}_shape"
                lines = []
                for line in path.read_text(encoding="utf-8").splitlines():
                    uid, _, shape = line.partition(" ")
                    # Already 2-D, so a rerun is idempotent.
                    lines.append(line if "," in shape else f"{uid} {shape},{vocab}")
                path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                logger.info(
                    "[%s] %s -> %d entries, vocab %d",
                    mode,
                    path.name,
                    len(lines),
                    vocab,
                )

    def _has_tokenizer(self) -> bool:
        tokenizer_config = self.training_config.tokenizer
        output_path = Path(tokenizer_config.save_path)
        model = output_path / f"{tokenizer_config.model_type}.model"
        tokens = output_path / "tokens.txt"
        return model.is_file() and tokens.is_file()

    def _special_symbols(self) -> List[str]:
        """Symbols SentencePiece must keep whole.

        Taken from ``tokenizer.nlsyms`` when the config names a builder, so a
        recipe owns its own symbol inventory; there is no sensible default,
        because the list depends on which languages and tasks the corpora emit.
        """
        tokenizer_config = self.training_config.tokenizer
        symbols = getattr(tokenizer_config, "nlsyms", None)
        if symbols is None:
            raise RuntimeError(
                "training_config.tokenizer.nlsyms must list the OWSM special "
                "symbols. Without them SentencePiece splits <eng> into "
                "'<', 'e', 'n', 'g', '>' and the task tokens never survive "
                "tokenization."
            )
        if isinstance(symbols, (str, os.PathLike)):
            path = Path(symbols)
            if not path.is_file():
                raise RuntimeError(f"tokenizer.nlsyms file not found: {path}")
            return path.read_text(encoding="utf-8").split()
        return [str(symbol) for symbol in symbols]

    def _training_text_path(self) -> Path:
        tokenizer_config = self.training_config.tokenizer
        configured = getattr(tokenizer_config, "train_file", None)
        if configured:
            return Path(configured)
        data_dir = getattr(self.training_config, "data_dir", None)
        if data_dir:
            return Path(data_dir) / "train_tokenizer" / "train.txt"
        return Path(tokenizer_config.save_path) / "train.txt"

    def _gather_text(self, train_text_path: Path) -> List[str]:
        """Return the tokenizer training text, reusing it when already built.

        The gather walks every corpus in the mixture and takes minutes to hours,
        so an existing file is reused rather than treated as an error. The ASR
        system raises instead -- and does it *after* the gather, so an
        interrupted run pays the whole cost again and then aborts. Set
        ``tokenizer.reuse_existing_text: false`` for the strict behaviour.
        """
        tokenizer_config = self.training_config.tokenizer
        if train_text_path.exists():
            if not getattr(tokenizer_config, "reuse_existing_text", True):
                raise RuntimeError(
                    f"Tokenizer training text already exists: {train_text_path}"
                )
            logger.info("Reusing tokenizer text at %s", train_text_path)
            return train_text_path.read_text(encoding="utf-8").splitlines()

        builder_config = getattr(tokenizer_config, "text_builder", None)
        target = builder_config.get("_target_") if builder_config else None
        if not target:
            raise RuntimeError(
                "training_config.tokenizer.text_builder._target_ must name the "
                "callable that builds tokenizer text."
            )
        logger.info("Building tokenizer text via %s", target)
        built = instantiate(builder_config, _convert_="all")

        if isinstance(built, (str, os.PathLike)):
            path = Path(built)
            if not path.is_file():
                raise RuntimeError(f"Tokenizer text file not found: {path}")
            texts = path.read_text(encoding="utf-8").splitlines()
        elif isinstance(built, Iterable):
            texts = [str(text) for text in built]
        else:
            raise RuntimeError(
                "text_builder must return a path or an iterable of strings "
                f"(got {type(built)})."
            )
        if not texts:
            raise RuntimeError(
                "text_builder returned no text. Check dataset preparation."
            )
        train_text_path.parent.mkdir(parents=True, exist_ok=True)
        train_text_path.write_text("\n".join(texts), encoding="utf-8")
        return texts

    def train_tokenizer(self, *args, **kwargs):
        """Train one SentencePiece model, reserving the OWSM special symbols."""
        self._reject_stage_args("train_tokenizer", args, kwargs)

        if self._has_tokenizer():
            logger.info("Tokenizer already exists. Skipping train_tokenizer().")
            return
        start = time.perf_counter()

        tokenizer_config = self.training_config.tokenizer
        symbols = self._special_symbols()
        vocab_size = int(tokenizer_config.vocab_size)
        if vocab_size <= len(symbols):
            raise RuntimeError(
                f"tokenizer.vocab_size ({vocab_size}) must exceed the "
                f"{len(symbols)} reserved symbols, which occupy the vocabulary "
                "before a single piece is learned."
            )

        train_text_path = self._training_text_path()
        texts = self._gather_text(train_text_path)
        output_path = Path(tokenizer_config.save_path)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Training %s vocab_size=%d on %d lines, reserving %d symbols",
            tokenizer_config.model_type,
            vocab_size,
            len(texts),
            len(symbols),
        )
        train_sentencepiece(
            train_text_path,
            output_path,
            vocab_size,
            character_coverage=getattr(tokenizer_config, "character_coverage", 1.0),
            model_type=tokenizer_config.model_type,
            user_defined_symbols=symbols,
        )
        logger.info(
            "Tokenizer training completed in %.2fs", time.perf_counter() - start
        )
