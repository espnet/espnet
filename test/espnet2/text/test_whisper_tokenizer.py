import sys

import pytest

from espnet2.text.whisper_tokenizer import OpenAIWhisperTokenizer

pytest.importorskip("whisper")

is_python_3_8_plus = sys.version_info >= (3, 8)


@pytest.mark.skipif(not is_python_3_8_plus)
@pytest.fixture(params=["whisper_multilingual"])
def whisper_tokenizer(request):
    return OpenAIWhisperTokenizer(request.param)


def test_init_en():
    tokenizer = OpenAIWhisperTokenizer("whisper_en")
    assert tokenizer.tokenizer.tokenizer.vocab_size == 50257


def test_init_invalid():
    with pytest.raises(ValueError):
        OpenAIWhisperTokenizer("whisper_aaa")


def test_repr(whisper_tokenizer: OpenAIWhisperTokenizer):
    print(whisper_tokenizer)


def test_tokenization_consistency(whisper_tokenizer: OpenAIWhisperTokenizer):
    s = "Hi, today's weather is nice. Hmm..."

    assert s == whisper_tokenizer.tokens2text(whisper_tokenizer.text2tokens(s))
