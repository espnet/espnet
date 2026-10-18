"""Tokenizer text helpers for the AmericasNLP 2022 recipe."""

from __future__ import annotations

from pathlib import Path
from typing import List

from egs3.americasnlp22.esp2_asr.dataset.builder import (
    resolve_language_dir,
    resolve_source_root,
)


def gather_training_text(
    lang: str,
    recipe_dir: str | Path | None = None,
    source_dir: str | Path | None = None,
    split: str = "train",
) -> List[str]:
    """Collect transcript text of one language for tokenizer training.

    Mirrors the ESPnet2 recipe's ``--bpe_train_text data/train_<lang>/text``:
    the SentencePiece model is trained on the ``source_raw`` transcripts of
    the single target language, unmodified. The ``train_tokenizer`` stage
    calls it through ``tokenizer.text_builder`` in ``conf/training.yaml``,
    passing every key of that block except ``func`` as a keyword argument.

    Args:
        lang: ISO language code, e.g. ``bzd``.
        recipe_dir: Recipe root used to resolve the local download directory.
            When omitted, the recipe directory of this module is used.
        source_dir: Optional corpus root override.
        split: Corpus split to read transcripts from.

    Returns:
        Transcript strings for tokenizer training, in ``meta.tsv`` order.

    Raises:
        FileNotFoundError: If the split directory does not exist.
        RuntimeError: If no transcript text is found.

    Examples:
        Wiring in ``conf/training.yaml``, where ``lang`` comes from the
        per-language config (e.g. ``conf/tuning/training_tav.yaml``):

        .. code-block:: yaml

            tokenizer:
              vocab_size: 100
              model_type: unigram
              save_path: ${data_dir}/${lang}_unigram100
              train_file: ${tokenizer.save_path}/train.txt
              text_builder:
                func: src.tokenizer.gather_training_text
                recipe_dir: ${recipe_dir}
                lang: ${lang}
                split: train

        Calling it directly from the recipe root, after ``create_dataset``:

        >>> texts = gather_training_text(lang="tav", recipe_dir=".")
        >>> len(texts)
        1419
        >>> for line in texts[:3]:
        ...     print(line)
        ahsi mUhano niaye tUle
        o, ~so'opUpe'le du'se
        no'opUnu U'mUano ~sayano, ~sayanonaha buupe

        ``python run.py --stages train_tokenizer --training_config
        conf/tuning/training_tav.yaml`` writes these lines to ``train.txt``
        and trains the SentencePiece model next to it:

        .. code-block:: text

            data/tav_unigram100/
            ├── train.txt            # the returned lines, one per line
            ├── unigram.model        # model_type=unigram, vocab_size=100
            ├── unigram.vocab
            ├── tokens.txt           # <blank>, <unk>, the pieces, <sos/eos>
            └── train_tokenizer.log

        ``tokens.txt``, the model's ``token_list``, starts with:

        .. code-block:: text

            <blank>
            <unk>
            ▁
            a
            '
    """
    recipe_root = (
        Path(recipe_dir).resolve()
        if recipe_dir is not None
        else Path(__file__).resolve().parents[1]
    )
    split_path = resolve_language_dir(
        resolve_source_root(recipe_root, source_dir=source_dir), lang
    ) / str(split)
    if not split_path.is_dir():
        raise FileNotFoundError(f"Split not found for tokenizer text: {split_path}")

    texts = []
    meta_path = split_path / "meta.tsv"
    with meta_path.open("r", encoding="utf-8") as fh:
        fh.readline()  # header
        for raw_line in fh:
            columns = raw_line.rstrip("\n").split("\t")
            if len(columns) < 3:
                continue
            text = columns[2].strip()
            if text:
                texts.append(text)
    if not texts:
        raise RuntimeError(f"No transcript text found under: {meta_path}")
    return texts
