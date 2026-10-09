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


def test_remove_non_linguistic_symbols_keeps_one_space():
    tokenizer = CharTokenizer(
        non_linguistic_symbols=["<noise>"], remove_non_linguistic_symbols=True
    )
    expected = tokenizer.text2tokens("hello world")
    assert tokenizer.text2tokens("hello <noise> world") == expected
    assert tokenizer.text2tokens("<noise> hello world <noise>") == expected
    assert tokenizer.text2tokens("hello <noise><noise> world") == expected
    # A symbol inside a word leaves no space behind.
    assert tokenizer.text2tokens("hel<noise>lo world") == expected
