import pytest

from espnet2.text.word_tokenizer import WordTokenizer


@pytest.fixture(params=[None, " "])
def word_tokenizer(request):
    return WordTokenizer(delimiter=request.param)


def test_repr(word_tokenizer: WordTokenizer):
    print(word_tokenizer)


def test_text2tokens(word_tokenizer: WordTokenizer):
    assert word_tokenizer.text2tokens("Hello World!! Ummm") == [
        "Hello",
        "World!!",
        "Ummm",
    ]


def test_tokens2text(word_tokenizer: WordTokenizer):
    assert word_tokenizer.tokens2text("Hello World!!".split()) == "Hello World!!"
