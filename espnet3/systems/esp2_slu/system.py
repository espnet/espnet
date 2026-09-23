"""SLU system: reserved intent labels and a second, transcript token list."""

from __future__ import annotations

import logging
from importlib import import_module
from pathlib import Path
from typing import Callable, Iterable, List

from espnet3.systems.asr.system import ASRSystem
from espnet3.systems.asr.tokenizers.sentencepiece import train_sentencepiece

logger = logging.getLogger(__name__)


def _load_hook(func_path: str) -> Callable:
    """Import a ``module.function`` hook named by a config field."""
    module_path, func_name = func_path.rsplit(".", 1)
    return getattr(import_module(module_path), func_name)


def _call_hook(hook_config, what: str) -> List[str]:
    """Call a ``{func: ..., **kwargs}`` config block and return its strings.

    Args:
        hook_config: The config block, whose ``func`` is a dotted path and
            whose remaining keys are that function's keyword arguments.
        what: What the hook produces, used in error messages.

    Returns:
        The hook's return value as a list of strings.

    Raises:
        RuntimeError: If ``func`` is unset or the hook returns nothing.
    """
    func_path = getattr(hook_config, "func", None) if hook_config else None
    if not func_path:
        raise RuntimeError(f"{what}.func must be set.")
    hook = _load_hook(func_path)
    kwargs = {key: value for key, value in hook_config.items() if key != "func"}
    values = [str(value) for value in hook(**kwargs)]
    if not values:
        raise RuntimeError(f"{what} returned nothing. Check dataset preparation.")
    return values


def write_word_token_list(token_list_path: str | Path, texts: Iterable[str]) -> Path:
    """Write the word vocabulary of ``texts``, framed by the special symbols.

    The layout ``espnet2.bin.tokenize_text --token_type word`` produces, which
    is what stage 5 of ``egs2/TEMPLATE/slu1/slu.sh`` builds for the transcript
    field. Writing is atomic, so an interrupted run cannot leave a half-written
    list behind for the next one to read.

    Args:
        token_list_path: Where to write the list, one token per line.
        texts: The lines whose words make up the vocabulary.

    Returns:
        The path written.
    """
    words: set[str] = set()
    for text in texts:
        words.update(text.split())

    tokens = ["<blank>", "<unk>", *sorted(words), "<sos/eos>"]
    output_path = Path(token_list_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)
    return output_path


class Esp2SluSystem(ASRSystem):
    """ASRSystem for ``espnet2.tasks.slu.SLUTask``'s two extra requirements.

    ``SLUTask`` predicts the intent as a label prefixed to the transcript, and
    optionally reads that transcript as a second input. Two things follow that
    ``ASRSystem`` cannot express, and both are handled in ``train_tokenizer``:

    * **The intent labels must stay whole.** Split into subwords, a label
      becomes several decoding steps that can each go wrong, and the
      first-token rule that intent scoring relies on no longer holds.
      SentencePiece reserves symbols for exactly this, but
      ``ASRSystem.train_tokenizer`` does not forward ``user_defined_symbols``.
    * **The transcript field needs its own token list**, a plain word
      vocabulary that ``ESPnetSLUModel`` detokenizes through before calling the
      Hugging Face tokenizer. That is a second artifact of the same stage, the
      way stage 5 of ``egs2/TEMPLATE/slu1/slu.sh`` builds both lists together.

    Where the labels and the text come from is the recipe's business, so both
    arrive through ``func`` hooks, as ``tokenizer.text_builder`` already does.

    Stages: overrides ``train_tokenizer`` and ``train``; every other stage is
    inherited.

    Config:
        Under ``training_config.tokenizer``, in addition to what ``ASRSystem``
        reads (``save_path``, ``vocab_size``, ``model_type``, ``text_builder``,
        and the optional ``character_coverage`` and ``train_file``)::

            tokenizer:
              user_defined_symbols_builder:
                func: src.tokenizer.read_intent_labels   # -> list[str]
                recipe_dir: ${recipe_dir}
              transcript_token_list:
                path: ${data_dir}/manifest/transcript_tokens.txt
                text_builder:
                  func: src.tokenizer.gather_transcript_text   # -> list[str]
                  recipe_dir: ${recipe_dir}

        ``vocab_size`` counts the reserved labels, so it must leave room for
        them on top of the subword inventory. Both blocks are optional: with
        neither set this behaves exactly as ``ASRSystem``.
    """

    def train(self, *args, **kwargs):
        """Train the model, writing the transcript token list first.

        ``ASRSystem.train`` only calls ``train_tokenizer`` when the
        SentencePiece model is missing, and a two-pass config typically points
        ``tokenizer.save_path`` at a tokenizer an earlier config already
        trained, so it never is. Building the list here as well is what makes
        ``--stages train`` work on its own; without it the stage fails on a
        missing ``transcript_token_list`` unless ``train_tokenizer`` happened
        to be run explicitly first.

        ``write_transcript_token_list`` returns early once the file exists, so
        reaching it from both entry points costs nothing.
        """
        self.write_transcript_token_list()
        return super().train(*args, **kwargs)

    def train_tokenizer(self, *args, **kwargs):
        """Train SentencePiece with the corpus intent labels reserved.

        Writes the model, vocabulary and ``tokens.txt`` under
        ``training_config.tokenizer.save_path``; re-running is a no-op once
        those exist. The transcript token list is written first and on every
        call, for the reason :meth:`train` describes.

        Raises:
            RuntimeError: If a configured hook is missing its ``func`` or
                returns nothing, or if the tokenizer training text already
                exists from an earlier interrupted run.
        """
        self._reject_stage_args("train_tokenizer", args, kwargs)

        self.write_transcript_token_list()

        if self._has_tokenizer():
            logger.info("Tokenizer already exists. Skipping train_tokenizer().")
            return

        tokenizer_config = self.training_config.tokenizer
        texts = _call_hook(
            getattr(tokenizer_config, "text_builder", None),
            "training_config.tokenizer.text_builder",
        )

        symbols_config = getattr(tokenizer_config, "user_defined_symbols_builder", None)
        user_defined_symbols = (
            _call_hook(
                symbols_config,
                "training_config.tokenizer.user_defined_symbols_builder",
            )
            if symbols_config is not None
            else []
        )

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
            "Training %s tokenizer on %d lines, reserving %d symbols",
            tokenizer_config.model_type,
            len(texts),
            len(user_defined_symbols),
        )
        train_sentencepiece(
            train_text_path,
            save_path,
            tokenizer_config.vocab_size,
            character_coverage=float(
                getattr(tokenizer_config, "character_coverage", 1.0) or 1.0
            ),
            model_type=tokenizer_config.model_type,
            user_defined_symbols=user_defined_symbols,
        )

    def write_transcript_token_list(self) -> Path | None:
        """Write the transcript token list, if this config asks for one.

        Returns:
            The path written or already present, or ``None`` when
            ``training_config.tokenizer.transcript_token_list`` is unset, which
            is the case for a one-pass config.

        Raises:
            RuntimeError: If the block is set but carries no ``path``, or if
                its text hook is misconfigured or returns nothing.
        """
        list_config = getattr(
            self.training_config.tokenizer, "transcript_token_list", None
        )
        if list_config is None:
            return None

        token_list_path = getattr(list_config, "path", None)
        if not token_list_path:
            raise RuntimeError(
                "training_config.tokenizer.transcript_token_list.path must be set "
                "when the block is present."
            )
        output_path = Path(token_list_path)
        if output_path.is_file():
            return output_path

        texts = _call_hook(
            getattr(list_config, "text_builder", None),
            "training_config.tokenizer.transcript_token_list.text_builder",
        )
        logger.info("Building transcript token list at %s", output_path)
        return write_word_token_list(output_path, texts)

    def _resolve_train_text_path(self, save_path: Path) -> Path:
        """Return where the tokenizer training text goes, as the base stage does."""
        train_file = getattr(self.training_config.tokenizer, "train_file", None)
        if train_file:
            return Path(train_file)
        data_dir = getattr(self.training_config, "data_dir", None)
        if data_dir:
            return Path(data_dir) / "train_tokenizer" / "train.txt"
        return save_path / "train.txt"
