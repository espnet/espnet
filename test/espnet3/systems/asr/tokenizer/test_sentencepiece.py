"""Tests for sentencepiece tokenizer utilities."""

from pathlib import Path

import pytest
import torch

from espnet2.text.sentencepiece_tokenizer import SentencepiecesTokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet3.systems.asr.tokenizer.sentencepiece import (
    add_special_tokens,
    prepare_sentences,
    train_sentencepiece,
)


def _test_data_dir():
    """Return the shared sentencepiece test data directory."""
    return Path(__file__).parents[5] / "test_utils" / "espnet3" / "sentencepiece_model"


def test_prepare_sentences_joins_dump_files(tmp_path):
    """Ensure prepare_sentences concatenates dump files into train.txt."""
    data_dir = _test_data_dir()
    output_dir = tmp_path / "spm"
    prepare_sentences(
        [data_dir / "dump1.txt", data_dir / "dump2.txt"],
        output_dir,
    )

    train_txt = output_dir / "train.txt"
    assert train_txt.is_file()
    assert train_txt.read_text(encoding="utf-8") == (
        "hello world\nthis is a test\nespnet3 sentencepiece"
    )


def test_train_sentencepiece_and_add_special_tokens(tmp_path, monkeypatch):
    """Ensure training outputs model/tokens and adding specials updates artifacts."""
    pytest.importorskip("sentencepiece")
    data_dir = _test_data_dir()
    output_dir = tmp_path / "model"

    monkeypatch.chdir(tmp_path)
    text = (data_dir / "train.txt").read_text(encoding="utf-8").replace("\n", "")
    vocab_size = len(set(text))
    train_sentencepiece(
        data_dir / "train.txt",
        output_dir,
        vocab_size=vocab_size + 3,
        model_type="char",
    )

    model_path = output_dir / "char.model"
    tokens_path = output_dir / "tokens.txt"
    assert model_path.is_file()
    assert tokens_path.is_file()

    tokens = tokens_path.read_text(encoding="utf-8").splitlines()
    assert tokens[0] == "<blank>"
    assert tokens[1] == "<unk>"
    assert tokens[-1] == "<sos/eos>"

    tokenizer = SentencepiecesTokenizer(model_path)
    converter = TokenIDConverter(tokens, "<unk>")
    embedding = torch.nn.Embedding(len(tokens), 4)

    new_tokenizer, new_converter, new_embedding = add_special_tokens(
        tokenizer,
        converter,
        embedding,
        ["<intent>", "<speaker>"],
        insert_after="<unk>",
    )

    assert new_tokenizer.model.endswith("_sp.model")
    assert Path(new_tokenizer.model).is_file()
    assert new_embedding.num_embeddings == len(tokens) + 2
    insert_at = new_converter.token_list.index("<unk>") + 1
    assert new_converter.token_list[insert_at : insert_at + 2] == [
        "<intent>",
        "<speaker>",
    ]

    new_tokenizer2, new_converter2, new_embedding2 = add_special_tokens(
        tokenizer,
        converter,
        embedding,
        ["<domain>"],
    )
    assert new_embedding2.num_embeddings == len(tokens) + 1
    assert new_converter2.token_list[-1] == "<domain>"
