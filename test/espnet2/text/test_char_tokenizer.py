import pytest

from espnet2.text.char_tokenizer import CharTokenizer


@pytest.fixture
def char_tokenizer():
    return CharTokenizer(non_linguistic_symbols=["[foo]"])


def test_repr(char_tokenizer: CharTokenizer):
    print(char_tokenizer)


def test_text2tokens(char_tokenizer: CharTokenizer):
    assert char_tokenizer.text2tokens("He[foo]llo") == [
        "H",
        "e",
        "[foo]",
        "l",
        "l",
        "o",
    ]


def test_token2text(char_tokenizer: CharTokenizer):
    assert char_tokenizer.tokens2text(["a", "b", "c"]) == "abc"
