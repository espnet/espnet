"""ST system: separate source and target vocabularies, as egs2's st.sh builds."""

import logging
import os
from importlib import import_module
from pathlib import Path
from typing import Iterable, List

from omegaconf import DictConfig

from espnet3.systems.base.system import BaseSystem
from espnet3.systems.esp2_st.tokenizers.sentencepiece import train_sentencepiece

logger = logging.getLogger(__name__)

_SIDES = ("tgt", "src")


class STSystem(BaseSystem):
    """System for speech translation, with one vocabulary per side.

    ``espnet2.tasks.st.STTask`` carries two vocabularies, not one:

    * the **target** side feeds the ST decoder (``criterion_st``,
      ``size=vocab_size``) and the ST-CTC;
    * the **source** side feeds the auxiliary ASR decoder
      (``criterion_asr``, ``size=src_vocab_size``) and the ASR-CTC.

    They are independent -- different vocabulary sizes, and in egs2 different
    casing conventions too (``tgt_case=tc`` truecased, ``src_case=lc.rm``
    lowercased and stripped of punctuation). ``ASRSystem`` trains a single
    model and so cannot express this; this subclass trains one per side.

    Config shape under ``tokenizer``::

        tokenizer:
          tgt:
            vocab_size: 4000
            model_type: bpe
            save_path: ${data_dir}/bpe_tgt_4000
            text_builder: {func: ..., ...}
          src:
            vocab_size: 4000
            model_type: bpe
            save_path: ${data_dir}/bpe_src_4000
            text_builder: {func: ..., ...}

    The model config then points at the two token lists::

        model:
          token_list:     ${tokenizer.tgt.save_path}/tokens.txt
          src_token_list: ${tokenizer.src.save_path}/tokens.txt

    A side may be omitted, in which case it is skipped -- egs2's
    ``use_src_lang=false`` trains the target model only.
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
        """Initialize the ST system with optional stage configs.

        Args:
            training_config: Training configuration.
            inference_config: Inference configuration.
            metrics_config: Measurement configuration.
            publication_config: Publication configuration for the model
                packing and upload stages.
            stage_log_mapping: Optional per-stage log directory overrides.
            demo_config: Demo configuration for the demo stages.
        """
        # ASRSystem points train_tokenizer's log at tokenizer.save_path. There
        # is no single save_path here -- the sides live under tokenizer.tgt and
        # tokenizer.src -- so the target side stands for the stage.
        super().__init__(
            training_config=training_config,
            inference_config=inference_config,
            metrics_config=metrics_config,
            publication_config=publication_config,
            stage_log_mapping={
                "train_tokenizer": "training_config.tokenizer.tgt.save_path",
                **(stage_log_mapping or {}),
            },
            demo_config=demo_config,
        )

    def train(self, *args, **kwargs):
        """Train the model, training any missing tokenizer first.

        Raises:
            RuntimeError: If neither ``dataset`` nor ``dataset_dir`` is set on
                the training config.
        """
        self._reject_stage_args("train", args, kwargs)
        logger.info("STSystem.train(): starting training process")

        dataset_dir = getattr(self.training_config, "dataset_dir", None)
        dataset_config = getattr(self.training_config, "dataset", None)
        if dataset_dir is None and dataset_config is None:
            raise RuntimeError(
                "training_config.dataset or training_config.dataset_dir must be "
                "set for training."
            )

        # False when EITHER configured side is missing, so a run that added the
        # source side after training the target one fills in just that side.
        if not self._has_tokenizer():
            self.train_tokenizer()

        return super().train()

    def _has_tokenizer_side(self, side_config) -> bool:
        model = Path(side_config.save_path) / f"{side_config.model_type}.model"
        tokens = Path(side_config.save_path) / "tokens.txt"
        return model.is_file() and tokens.is_file()

    def _has_tokenizer(self) -> bool:
        """Whether every configured side already has a trained model.

        ``train`` calls this before training and runs ``train_tokenizer``
        when it is False. It is spelled per side because a two-vocabulary
        config has no single ``tokenizer.save_path``: the sides live under
        ``tokenizer.tgt`` and ``tokenizer.src``, and reading the flat key
        would abort the train stage before its first step with::

            omegaconf.errors.ConfigAttributeError: Missing key save_path
                full_key: tokenizer.save_path

        Returning False when a configured side is missing is what makes
        ``train`` fall through to ``train_tokenizer``, which then skips
        whichever sides are already built.
        """
        config = self.training_config.tokenizer
        sides = [s for s in _SIDES if getattr(config, s, None) is not None]
        if not sides:
            return False
        return all(self._has_tokenizer_side(getattr(config, s)) for s in sides)

    def _gather_texts(self, side: str, side_config) -> tuple[Path, List[str]]:
        train_path = Path(
            getattr(side_config, "train_file", "")
            or f"{self.training_config.data_dir}/train_tokenizer/{side}.txt"
        )
        reuse = bool(getattr(side_config, "reuse_existing_text", True))
        if train_path.exists():
            if not reuse:
                raise RuntimeError(
                    f"Tokenizer training text already exists: {train_path}"
                )
            logger.info("[%s] reusing existing tokenizer text at %s", side, train_path)
            return train_path, train_path.read_text(encoding="utf-8").splitlines()

        builder_config = side_config.text_builder
        module_path, func_name = builder_config.func.rsplit(".", 1)
        builder = getattr(import_module(module_path), func_name)
        built = builder(**{k: v for k, v in builder_config.items() if k != "func"})
        if isinstance(built, (str, os.PathLike)):
            texts = Path(built).read_text(encoding="utf-8").splitlines()
        elif isinstance(built, Iterable):
            texts = [str(text) for text in built]
        else:
            raise RuntimeError(
                f"[{side}] tokenizer text builder must return a path or iterable"
            )
        if not texts:
            raise RuntimeError(f"[{side}] tokenizer text builder returned no text")
        train_path.parent.mkdir(parents=True, exist_ok=True)
        train_path.write_text("\n".join(texts), encoding="utf-8")
        return train_path, texts

    def collect_stats(self, *args, **kwargs):
        """Collect stats, then make the token shape files 2-D.

        ``espnet2``'s ``st.sh`` bounds ``batch_bins`` over THREE shape files --
        ``speech_shape`` plus one per text stream -- and writes the text ones as
        ``"L,V"`` by appending the vocabulary size, so one token costs ``V``
        bins. Without that the batcher cannot see text length at all and a batch
        of short utterances grows until it exhausts GPU memory.

        ``collect_stats`` records the token streams as 1-D lengths; only the
        System knows which vocabulary belongs to which stream, so the vocabulary
        size is appended here. ``text`` takes the target vocabulary and
        ``src_text`` the source one, matching ``STTask``.
        """
        self._reject_stage_args("collect_stats", args, kwargs)
        result = super().collect_stats()
        self._append_vocab_size_to_text_shapes()
        return result

    def _vocab_size(self, side: str) -> int:
        """Vocabulary size for one side, as st.sh takes it: token-list lines."""
        side_config = getattr(self.training_config.tokenizer, side)
        tokens = Path(side_config.save_path) / "tokens.txt"
        with tokens.open(encoding="utf-8") as stream:
            return sum(1 for _ in stream)

    def _append_vocab_size_to_text_shapes(self) -> None:
        """Rewrite ``<stream>_shape`` from ``L`` to ``L,V`` in place."""
        stats_dir = Path(self.training_config.stats_dir)
        streams = {"text": "tgt", "src_text": "src"}
        for mode in ("train", "valid"):
            for stream, side in streams.items():
                if getattr(self.training_config.tokenizer, side, None) is None:
                    continue
                path = stats_dir / mode / f"{stream}_shape"
                if not path.is_file():
                    continue
                vocab = self._vocab_size(side)
                lines = []
                for line in path.read_text(encoding="utf-8").splitlines():
                    uid, _, shape = line.partition(" ")
                    if "," in shape:  # already 2-D; rerun is idempotent
                        lines.append(line)
                    else:
                        lines.append(f"{uid} {shape},{vocab}")
                path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                logger.info(
                    "[%s] %s -> %s entries, vocab %d",
                    mode,
                    path.name,
                    len(lines),
                    vocab,
                )

    def train_tokenizer(self, *args, **kwargs):
        """Train the target and source SentencePiece models.

        Expected config -- one block per side, each side optional except
        ``tgt``::

            tokenizer:
              tgt:
                vocab_size: 4000
                model_type: bpe
                save_path: ${data_dir}/bpe_tgt_4000
                text_builder:
                  func: egs3.must_c.esp2_st.dataset.gather_training_text
                  recipe_dir: ${recipe_dir}
                  tgt_lang: de
              src:
                vocab_size: 4000
                model_type: bpe
                save_path: ${data_dir}/bpe_src_4000
                text_builder: {...}

        ``text_builder.func`` is a dotted path to a callable returning either a
        path or an iterable of lines; every other key under it is passed
        through as a keyword argument. Set ``train_file`` to read prepared text
        instead, and ``reuse_existing_text: false`` to make a leftover file an
        error rather than silently reused.

        What it writes, per side::

            data/bpe_tgt_4000/bpe.model     SentencePiece model
            data/bpe_tgt_4000/bpe.vocab     SentencePiece vocabulary
            data/bpe_tgt_4000/tokens.txt    one token per line, the token_list
            data/train_tokenizer/tgt.txt    the gathered training text

        ``tokens.txt`` is what the model config points ``token_list`` and
        ``src_token_list`` at, and its line count is the vocabulary size
        ``collect_stats`` appends to the text shapes.

        A side whose model and ``tokens.txt`` already exist is skipped, so a
        rerun after adding the source side trains only the missing one.

        Raises:
            RuntimeError: If no side is configured, or a side's builder
                produces no text.
        """
        self._reject_stage_args("train_tokenizer", args, kwargs)
        config = self.training_config.tokenizer

        sides = [s for s in _SIDES if getattr(config, s, None) is not None]
        if not sides:
            raise RuntimeError(
                "STSystem expects tokenizer.tgt (and optionally tokenizer.src). "
                "For a single shared vocabulary use ASRSystem."
            )

        for side in sides:
            side_config = getattr(config, side)
            if self._has_tokenizer_side(side_config):
                logger.info("[%s] tokenizer already present, skipping", side)
                continue
            train_path, texts = self._gather_texts(side, side_config)
            if not texts:
                raise RuntimeError(f"[{side}] tokenizer training text is empty")
            logger.info(
                "[%s] training %s vocab_size=%s on %d lines",
                side,
                side_config.model_type,
                side_config.vocab_size,
                len(texts),
            )
            train_sentencepiece(
                train_path,
                side_config.save_path,
                side_config.vocab_size,
                character_coverage=getattr(side_config, "character_coverage", 1.0),
                model_type=side_config.model_type,
                user_defined_symbols=list(
                    getattr(side_config, "user_defined_symbols", None) or []
                ),
            )
