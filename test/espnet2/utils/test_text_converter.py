from pathlib import Path
import string

import numpy as np
import pytest
import sentencepiece as spm
from typing import List

from espnet2.utils.text_converter import build_text_converter
from espnet2.utils.text_converter import Text2CharsConverter
from espnet2.utils.text_converter import Text2SentencepiecesConverter
from espnet2.utils.text_converter import Text2WordsConverter


@pytest.fixture(params=[None, " "])
def word_converter(request):
    return Text2WordsConverter(["<unk>", "Hello", "World!!"], delimiter=request.param)


@pytest.fixture
def char_converter():
    return Text2CharsConverter(["<unk>"] + list(string.printable))


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
    return Text2SentencepiecesConverter(model=model, token_list=token_list)


def test_Text2Sentencepieces_repr(spm_converter: Text2SentencepiecesConverter):
    print(spm_converter)


def test_Text2Sentencepieces(spm_converter: Text2SentencepiecesConverter):
    ids = [1, 2, 0, 1, 2, 8, 10]
    assert spm_converter.tokens2ids(spm_converter.ids2tokens(ids)) == ids


def test_Text2Sentencepieces_text2ids(spm_converter: Text2SentencepiecesConverter,):
    assert spm_converter.ids2text(spm_converter.text2ids("Hello")) == "Hello"


def test_Text2Sentencepieces_text2tokens(spm_converter: Text2SentencepiecesConverter,):
    assert spm_converter.tokens2text(spm_converter.text2tokens("Hello")) == "Hello"


def test_Text2Words_repr(word_converter: Text2WordsConverter):
    print(word_converter)


def test_Text2Words_ids2tokens(word_converter: Text2WordsConverter):
    assert word_converter.ids2tokens([1, 2, 0]) == [
        "Hello",
        "World!!",
        "<unk>",
    ]


def test_Text2Words_tokens2ids(word_converter: Text2WordsConverter):
    assert word_converter.tokens2ids("Hello World!! Umm".split()) == [1, 2, 0]


def test_Text2Words_text2ids(word_converter: Text2WordsConverter):
    assert word_converter.text2ids("Hello World!! Ummm") == [1, 2, 0]


def test_Text2Words_text2tokens(word_converter: Text2WordsConverter):
    assert word_converter.text2tokens("Hello World!! Ummm") == [
        "Hello",
        "World!!",
        "Ummm",
    ]


def test_Text2Words_ids2text(word_converter: Text2WordsConverter):
    assert word_converter.ids2text([1, 2]) == "Hello World!!"


def test_Text2Words_tokens2text(word_converter: Text2WordsConverter):
    assert word_converter.tokens2text("Hello World!!".split()) == "Hello World!!"


def test_Text2Chars_repr(char_converter: Text2CharsConverter):
    print(char_converter)


def test_Text2Chars_ids2tokens(char_converter: Text2CharsConverter):
    assert char_converter.ids2tokens([1, 2, 0]) == ["0", "1", "<unk>"]


def test_Text2Chars_text2ids(char_converter: Text2CharsConverter):
    assert char_converter.text2ids("Hello") == [44, 15, 22, 22, 25]


def test_Text2Chars_text2tokens(char_converter: Text2CharsConverter):
    assert char_converter.text2tokens("Hello") == ["H", "e", "l", "l", "o"]


def test_Text2Chars_tokens2ids(char_converter: Text2CharsConverter):
    assert char_converter.tokens2ids("Hello") == [44, 15, 22, 22, 25]


def test_Text2Chars_ids2text(char_converter: Text2CharsConverter):
    assert char_converter.ids2text([1, 2]) == "01"


def test_Text2Chars_tokens2text(char_converter: Text2CharsConverter):
    assert char_converter.tokens2text("Hello") == "Hello"


def test_Text2Chars_nounk():
    with pytest.raises(RuntimeError):
        Text2CharsConverter(list(string.printable))


def test_Text2Chars_no_1dim_array(char_converter: Text2CharsConverter):
    with pytest.raises(ValueError):
        char_converter.ids2tokens(np.random.randn(2, 2))


def test_Text2Chars_get_num_vocaburary_size(char_converter: Text2CharsConverter,):
    assert char_converter.get_num_vocaburary_size() == 101


@pytest.mark.parametrize(
    "token_type, token_list",
    [
        ("char", ["<unk>"] + list(string.printable)),
        ("word", ["<unk>", "hee", "foo"]),
        ("bpe", None),
    ],
)
def test_build_text_converter(token_type: str, token_list: List[str], spm_srcs):
    model, vocabs = spm_srcs
    if token_list is None:
        token_list = vocabs

    build_text_converter(token_type=token_type, token_list=token_list, bpemodel=model)


def test_build_text_converter_without_model():
    with pytest.raises(ValueError):
        build_text_converter(token_type="bpe", token_list=())


def test_build_text_converter_unknown_type():
    with pytest.raises(ValueError):
        build_text_converter(token_type="ddd", token_list=())


def test_build_text_converter_duplicated():
    with pytest.raises(RuntimeError):
        build_text_converter(token_type="char", token_list=("unk", "e", "e"))
