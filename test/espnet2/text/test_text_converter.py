from pathlib import Path
import string

import pytest
import sentencepiece as spm

from espnet2.text.char_tokenizer import CharTokenizer
from espnet2.text.sentencepiece_tokenizer import SentencepiecesTokenizer
from espnet2.text.word_tokenizer import WordTokenizer


@pytest.fixture(params=[None, " "])
def word_converter(request):
    return WordTokenizer(delimiter=request.param)


@pytest.fixture
def char_converter():
    return CharTokenizer(["[foo]"])


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
def spm_converter(tmp_path, spm_srcs):
    model, vocabs = spm_srcs
    sp = spm.SentencePieceProcessor()
    sp.load(model)

    token_list = tmp_path / "token.list"
    with token_list.open("w") as f:
        for v in vocabs:
            f.write(f"{v}\n")
    return SentencepiecesTokenizer(model=model)


def test_Text2Sentencepieces_repr(spm_converter: SentencepiecesTokenizer):
    print(spm_converter)


def test_Text2Sentencepieces_text2tokens(spm_converter: SentencepiecesTokenizer):
    assert spm_converter.tokens2text(spm_converter.text2tokens("Hello")) == "Hello"


def test_Text2Words_repr(word_converter: WordTokenizer):
    print(word_converter)


def test_Text2Words_text2tokens(word_converter: WordTokenizer):
    assert word_converter.text2tokens("Hello World!! Ummm") == [
        "Hello",
        "World!!",
        "Ummm",
    ]


def test_Text2Words_tokens2text(word_converter: WordTokenizer):
    assert word_converter.tokens2text("Hello World!!".split()) == "Hello World!!"


def test_Text2Chars_repr(char_converter: CharTokenizer):
    print(char_converter)


def test_Text2Chars_text2tokens(char_converter: CharTokenizer):
    assert char_converter.text2tokens("He[foo]llo") == [
        "H",
        "e",
        "[foo]",
        "l",
        "l",
        "o",
    ]
