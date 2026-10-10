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


@pytest.mark.execution_timeout(5)
@pytest.mark.parametrize("from_file", [True, False])
@pytest.mark.parametrize("remove", [True, False])
def test_empty_non_linguistic_symbol(tmp_path, from_file, remove):
    if from_file:
        symbols = tmp_path / "nlsyms.txt"
        symbols.write_text("[foo]\n\n \t\n", encoding="utf-8")
    else:
        symbols = ["[foo]", ""]
    tokenizer = CharTokenizer(
        non_linguistic_symbols=symbols, remove_non_linguistic_symbols=remove
    )
    expected = ["a", "b"] if remove else ["a", "[foo]", "b"]
    assert tokenizer.text2tokens("a[foo]b") == expected


@pytest.mark.execution_timeout(5)
def test_empty_nonsplit_symbol():
    tokenizer = CharTokenizer(nonsplit_symbols=["", "[foo]"])
    assert tokenizer.text2tokens("a[foo]b") == ["a", "[foo]", "b"]
