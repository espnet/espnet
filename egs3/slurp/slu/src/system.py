"""Recipe-local system that reserves SLURP's intent labels in the vocabulary."""

from __future__ import annotations

import logging
from importlib import import_module
from pathlib import Path

from egs3.slurp.slu.src.tokenizer import (
    build_transcript_token_list,
    read_intent_labels,
)
from espnet3.systems.asr.system import ASRSystem
from espnet3.systems.asr.tokenizers.sentencepiece import train_sentencepiece

logger = logging.getLogger(__name__)


class SLUSystem(ASRSystem):
    """ASR system whose tokenizer keeps every intent label a single token.

    The recipe trains a plain ASR model on ``"<intent> <transcript>"``, so the
    intent has to survive tokenization intact: split into subwords it becomes
    several inference steps that can each go wrong, and the first-token rule the
    metric relies on no longer holds. SentencePiece reserves symbols for exactly
    this, but :meth:`ASRSystem.train_tokenizer` does not forward
    ``user_defined_symbols`` to
    :func:`espnet3.systems.asr.tokenizers.sentencepiece.train_sentencepiece`,
    so this subclass overrides that one stage.

    Delete this class and point ``run.py`` back at :class:`ASRSystem` if the
    shared stage ever forwards the option itself; nothing else here differs.

    The same stage also writes the transcript token list the SLU configs
    need, the way stage 5 of ``egs2/TEMPLATE/slu1/slu.sh`` builds both lists
    together. It is a separate artifact from the SentencePiece model and is
    built even when that model is already there, because the SLU configs
    reuse the tokenizer the ASR config trained.

    Stages: overrides ``train_tokenizer``; every other stage is inherited.

    Config:
        Reads ``training_config.tokenizer`` -- ``save_path``, ``vocab_size``,
        ``model_type``, ``text_builder.func`` (plus that hook's own keyword
        arguments), and the optional ``character_coverage`` and ``train_file``.
        ``vocab_size`` counts the reserved labels, so it must leave room for
        them on top of the subword inventory.

        The optional ``transcript_token_list`` block turns on the second
        artifact: ``path`` plus any keyword argument of
        :func:`src.tokenizer.build_transcript_token_list`. The ASR configs
        leave it unset and the stage then behaves exactly as the base one.

    Examples:
        In ``run.py``::

            from egs3.slurp.slu.src.system import SLUSystem
            main(args=args, system_cls=SLUSystem, stages=stages_to_run)
    """

    def train_tokenizer(self, *args, **kwargs):
        """Train SentencePiece with the corpus intent labels reserved.

        Reads the label list ``create_dataset`` wrote, so that stage must have
        run first. Writes the model, vocabulary and ``tokens.txt`` under
        ``training_config.tokenizer.save_path``; re-running is a no-op once
        those exist. The transcript token list is written first and on every
        call, since the SLU configs point ``save_path`` at the tokenizer the
        ASR config already trained and would otherwise return before building
        it.

        Raises:
            RuntimeError: If ``tokenizer.text_builder.func`` is unset, if the
                hook returns no text, or if the tokenizer training text already
                exists from an earlier interrupted run.
            FileNotFoundError: If the intent label list is missing, or if a
                configured transcript source has not been filled in yet.
        """
        self._reject_stage_args("train_tokenizer", args, kwargs)

        self._write_transcript_token_list()

        if self._has_tokenizer():
            logger.info("Tokenizer already exists. Skipping train_tokenizer().")
            return

        tokenizer_config = self.training_config.tokenizer
        builder_config = getattr(tokenizer_config, "text_builder", None)
        if builder_config is None or not getattr(builder_config, "func", None):
            raise RuntimeError(
                "training_config.tokenizer.text_builder.func must be set to build "
                "tokenizer text."
            )

        module_path, func_name = builder_config.func.rsplit(".", 1)
        gather_text = getattr(import_module(module_path), func_name)
        builder_kwargs = {k: v for k, v in builder_config.items() if k != "func"}
        texts = [str(text) for text in gather_text(**builder_kwargs)]
        if not texts:
            raise RuntimeError(
                "Tokenizer text_builder returned no text. Check dataset preparation."
            )

        recipe_dir = builder_kwargs.get("recipe_dir", self.training_config.recipe_dir)
        intent_labels = read_intent_labels(recipe_dir)

        save_path = Path(tokenizer_config.save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        train_text_path = self._resolve_train_text_path(save_path)
        if train_text_path.exists():
            raise RuntimeError(
                f"Tokenizer training text already exists: {train_text_path}"
            )
        train_text_path.parent.mkdir(parents=True, exist_ok=True)
        train_text_path.write_text("\n".join(texts), encoding="utf-8")

        logger.info(
            "Training %s tokenizer on %d lines, reserving %d intent labels",
            tokenizer_config.model_type,
            len(texts),
            len(intent_labels),
        )
        train_sentencepiece(
            train_text_path,
            save_path,
            tokenizer_config.vocab_size,
            character_coverage=float(
                getattr(tokenizer_config, "character_coverage", 1.0) or 1.0
            ),
            model_type=tokenizer_config.model_type,
            user_defined_symbols=intent_labels,
        )

    def _write_transcript_token_list(self) -> Path | None:
        """Write the transcript token list, if this config asks for one.

        Returns:
            The path written or already present, or ``None`` when
            ``training_config.tokenizer.transcript_token_list`` is unset, which
            is the case for the ASR configs.

        Raises:
            RuntimeError: If the block is set but carries no ``path``.
        """
        list_config = getattr(
            self.training_config.tokenizer, "transcript_token_list", None
        )
        if list_config is None:
            return None

        build_kwargs = {k: v for k, v in list_config.items() if k != "path"}
        token_list_path = getattr(list_config, "path", None)
        if not token_list_path:
            raise RuntimeError(
                "training_config.tokenizer.transcript_token_list.path must be set "
                "when the block is present."
            )
        build_kwargs.setdefault("recipe_dir", self.training_config.recipe_dir)

        logger.info("Building transcript token list at %s", token_list_path)
        return build_transcript_token_list(token_list_path, **build_kwargs)

    def _resolve_train_text_path(self, save_path: Path) -> Path:
        """Return where the tokenizer training text goes, as the base stage does."""
        train_file = getattr(self.training_config.tokenizer, "train_file", None)
        if train_file:
            return Path(train_file)
        data_dir = getattr(self.training_config, "data_dir", None)
        if data_dir:
            return Path(data_dir) / "train_tokenizer" / "train.txt"
        return save_path / "train.txt"
