"""The ``create_token_list`` stage: build a token list from a manifest.

``F5TTSSystem.create_token_list`` is a thin method that calls
:func:`create_token_list` with the training config. The stage reads the TSV
manifest an F5-TTS recipe's ``create_dataset`` stage writes, with the
transcript in the third column.
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Iterator, Union

from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf

from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.cleaner import TextCleaner

logger = logging.getLogger(__name__)


def _get_required_config(config, key: str, error_message: str):
    """Return ``config[key]``, raising ``RuntimeError`` when it is missing.

    Same contract as ``BaseSystem._get_required_config``, kept here so the
    stage function can be called without a system instance.
    """
    value = config.get(key, None) if config is not None else None
    if value is None:
        raise RuntimeError(error_message)
    return value


def _iter_transcripts(manifest_path: Union[str, Path]) -> Iterator[str]:
    r"""Yield the transcript of every manifest row that has one.

    Rows are ``utt_id\twav_path\ttext[\tspeaker_id]``. Rows with fewer than
    three columns or an empty transcript are skipped, the same rows the
    ``remove_long_short`` stage drops.
    """
    with open(manifest_path, "r", encoding="utf-8") as manifest_file:
        for line in manifest_file:
            parts = line.rstrip().split("\t")
            if len(parts) > 2 and parts[2].strip():
                yield parts[2]


def create_token_list(config: DictConfig) -> None:
    """Write the token list of the training transcripts to a file.

    The body of the ``create_token_list`` stage. It reads the transcript
    column of the training manifest, cleans and tokenizes it with ESPnet2's
    text front end, and writes one token per line to
    ``save_path/filename``, most frequent first with the configured special
    symbols inserted. Re-running the stage overwrites the file.

    Configuration should include (under ``create_token_list``):

      - ``save_path``: Directory in which to save the token-list file.
      - ``filename``: Token-list file name, such as ``tokens.txt``.
      - ``manifest_path``: Training manifest. Defaults to
        ``data/manifest/train.tsv``, relative to the working directory.
      - ``token_type``: ``char``, ``word``, ``bpe`` or ``phn``. Defaults to
        ``char``.
      - ``cleaner``: Optional text-cleaner name, such as ``tacotron``.
      - ``g2p``: Optional grapheme-to-phoneme model name.
      - ``bpemodel``, ``delimiter``, ``space_symbol``,
        ``non_linguistic_symbols``, ``remove_non_linguistic_symbols``:
        Forwarded to ``espnet2.text.build_tokenizer.build_tokenizer``.
      - ``add_symbol`` / ``add_nonsplit_symbol``: Special symbols to insert,
        each written ``"<symbol>:<index>"``; a negative index counts from
        the end.
      - ``cutoff``: Drop tokens seen this many times or fewer. Defaults to
        ``0``, which keeps every token.
      - ``vocabulary_size``: Upper bound on the list length, special symbols
        included. ``0`` or a negative value means unlimited.
      - ``vocab_builder`` / ``vocab_builder_conf``: Dotted path of a callable
        ``builder(texts: list[str], **vocab_builder_conf) -> list[str]`` and
        its keyword arguments. When set, the callable receives the cleaned
        transcripts and returns the full ordered token list, replacing the
        frequency-count construction and the special-symbol handling above.

    Args:
        config: Training config holding the ``create_token_list`` block.

    Raises:
        RuntimeError: If the ``create_token_list`` block, ``save_path`` or
            ``filename`` is missing, the manifest does not exist,
            ``vocabulary_size`` is smaller than the number of ``add_symbol``
            entries, or a special symbol is not written ``"<symbol>:<index>"``.

    Examples:
        .. code-block:: yaml

            create_token_list:
              save_path: ${data_dir}/token_list
              filename: tokens.txt
              manifest_path: ${data_dir}/manifest_filtered/train.tsv
              token_type: char
              cleaner: tacotron
              add_symbol:
                - "<blank>:0"
                - "<unk>:1"
                - "<sos/eos>:-1"

        With a custom vocabulary builder, here F5-TTS's zh+en pinyin one:

        .. code-block:: yaml

            create_token_list:
              save_path: ${data_dir}/token_list
              filename: vocab.txt
              vocab_builder: espnet3.systems.f5tts.pinyin.build_pinyin_vocab

        .. code-block:: python

            >>> create_token_list(training_config)  # doctest: +SKIP
    """
    create_token_list_config = _get_required_config(
        config,
        "create_token_list",
        "training_config.create_token_list must be set for create_token_list stage.",
    )
    save_dir = Path(
        _get_required_config(
            create_token_list_config,
            "save_path",
            "training_config.create_token_list.save_path must be set "
            "for create_token_list stage.",
        )
    )
    filename = _get_required_config(
        create_token_list_config,
        "filename",
        "training_config.create_token_list.filename must be set "
        "(e.g. 'tokens.txt'); save_path is the output directory.",
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / filename

    manifest_path = Path(
        create_token_list_config.get("manifest_path", "data/manifest/train.tsv")
    ).resolve()
    if not manifest_path.exists():
        raise RuntimeError(
            f"Manifest file not found for token list creation: "
            f"{manifest_path}. Please ensure the manifest file is "
            "generated and the path is correct."
        )

    text_cleaner = TextCleaner(create_token_list_config.get("cleaner", None))

    # A custom vocabulary builder fully replaces the frequency-count and
    # special-symbol construction below, so any tokenizer can plug its own
    # vocabulary into this stage.
    vocab_builder_path = create_token_list_config.get("vocab_builder", None)
    if vocab_builder_path is not None:
        texts = [text_cleaner(text) for text in _iter_transcripts(manifest_path)]
        vocab_builder = get_method(path=vocab_builder_path)
        vocab_builder_config = (
            create_token_list_config.get("vocab_builder_conf", {}) or {}
        )
        if not isinstance(vocab_builder_config, dict):
            vocab_builder_config = OmegaConf.to_container(
                vocab_builder_config, resolve=True
            )
        tokens = vocab_builder(texts, **vocab_builder_config)

        with open(output_path, "w", encoding="utf-8") as output_file:
            for token in tokens:
                output_file.write(f"{token}\n")
        logger.info(
            "create_token_list: built %d tokens from %d transcripts via %s -> %s",
            len(tokens),
            len(texts),
            vocab_builder_path,
            output_path,
        )
        return

    # Defaults match espnet2's tokenizer front end where applicable.
    add_symbol = create_token_list_config.get("add_symbol", [])
    add_nonsplit_symbol = create_token_list_config.get("add_nonsplit_symbol", [])
    cutoff = create_token_list_config.get("cutoff", 0)
    vocabulary_size = create_token_list_config.get("vocabulary_size", 0)

    tokenizer = build_tokenizer(
        token_type=create_token_list_config.get("token_type", "char"),
        bpemodel=create_token_list_config.get("bpemodel", None),
        delimiter=create_token_list_config.get("delimiter", None),
        space_symbol=create_token_list_config.get("space_symbol", "<space>"),
        non_linguistic_symbols=create_token_list_config.get(
            "non_linguistic_symbols", None
        ),
        remove_non_linguistic_symbols=create_token_list_config.get(
            "remove_non_linguistic_symbols", False
        ),
        g2p_type=create_token_list_config.get("g2p", None),
        nonsplit_symbol=add_nonsplit_symbol,
    )

    counter = Counter()
    for text in _iter_transcripts(manifest_path):
        counter.update(tokenizer.text2tokens(text_cleaner(text)))

    # Most frequent first, dropping tokens at or below the cutoff.
    tokens_and_counts = [
        (token, count)
        for token, count in sorted(counter.items(), key=lambda item: -item[1])
        if count > cutoff
    ]

    # Restrict the vocabulary size
    if vocabulary_size > 0:
        if vocabulary_size < len(add_symbol):
            raise RuntimeError(f"vocabulary_size is too small: {vocabulary_size}")
        tokens_and_counts = tokens_and_counts[: vocabulary_size - len(add_symbol)]

    for symbol_and_id in add_symbol + add_nonsplit_symbol:
        # e.g symbol="<blank>:0"
        try:
            symbol, idx = symbol_and_id.split(":")
            idx = int(idx)
        except ValueError:
            raise RuntimeError(f"Format error: e.g. '<blank>:0': {symbol_and_id}")
        symbol = symbol.strip()

        # e.g. idx=0  -> append as the first symbol
        # e.g. idx=-1 -> append as the last symbol
        if idx < 0:
            idx = len(tokens_and_counts) + 1 + idx
        tokens_and_counts.insert(idx, (symbol, None))

    with open(output_path, "w", encoding="utf-8") as output_file:
        for token, _ in tokens_and_counts:
            output_file.write(f"{token}\n")

    total_count = sum(counter.values())
    in_vocabulary_count = sum(
        count for _, count in tokens_and_counts if count is not None
    )
    if total_count > 0:
        logger.info(
            "OOV rate = %.2f %%",
            (total_count - in_vocabulary_count) / total_count * 100,
        )
    else:
        logger.warning("create_token_list: manifest contained no tokens.")
