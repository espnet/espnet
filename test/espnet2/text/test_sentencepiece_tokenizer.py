import string
from pathlib import Path

import pytest
import sentencepiece as spm

from espnet2.text.sentencepiece_tokenizer import SentencepiecesTokenizer


@pytest.fixture
def spm_srcs(tmp_path: Path):
    input_text = tmp_path / "text"
    vocabsize = len(string.ascii_letters) + 4
    model_prefix = tmp_path / "model"
    model = str(model_prefix) + ".model"
    input_sentence_size = 100000

    with input_text.open("w") as f:
        f.write(string.ascii_letters + "\n")

    spm.SentencePieceTrainer.Train(
        f"--input={input_text} "
        f"--vocab_size={vocabsize} "
        f"--model_prefix={model_prefix} "
        f"--input_sentence_size={input_sentence_size}"
    )
    sp = spm.SentencePieceProcessor()
    sp.load(model)

    with input_text.open("r") as f:
        vocabs = {"<unk>", "▁"}
        for line in f:
            tokens = sp.DecodePieces(list(line.strip()))
        vocabs |= set(tokens)
    return model, vocabs


@pytest.fixture
def spm_tokenizer(tmp_path, spm_srcs):
    model, vocabs = spm_srcs
    sp = spm.SentencePieceProcessor()
    sp.load(model)

    token_list = tmp_path / "token.list"
    with token_list.open("w") as f:
        for v in vocabs:
            f.write(f"{v}\n")
    return SentencepiecesTokenizer(model=model)


def test_repr(spm_tokenizer: SentencepiecesTokenizer):
    print(spm_tokenizer)


def test_text2tokens(spm_tokenizer: SentencepiecesTokenizer):
    assert spm_tokenizer.tokens2text(spm_tokenizer.text2tokens("Hello")) == "Hello"
