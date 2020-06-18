import pytest

from espnet2.text.phoneme_tokenizer import PhonemeTokenizer

params = ["g2p_en"]
try:
    import pyopenjtalk

    params.extend(["pyopenjtalk", "pyopenjtalk_kana"])
    del pyopenjtalk
except ImportError:
    pass


@pytest.fixture(params=params)
def phoneme_tokenizer(request):
    return PhonemeTokenizer(g2p_type=request.param)


def test_repr(phoneme_tokenizer: PhonemeTokenizer):
    print(phoneme_tokenizer)


def test_text2tokens(phoneme_tokenizer: PhonemeTokenizer):
    if phoneme_tokenizer.g2p_type == "g2p_en":
        input = "Hello World"
        output = ["HH", "AH0", "L", "OW1", " ", "W", "ER1", "L", "D"]
    elif phoneme_tokenizer.g2p_type == "pyopenjtalk":
        input = "昔は俺も若かった"
        output = [
            "m",
            "u",
            "k",
            "a",
            "sh",
            "i",
            "w",
            "a",
            "o",
            "r",
            "e",
            "m",
            "o",
            "w",
            "a",
            "k",
            "a",
            "k",
            "a",
            "cl",
            "t",
            "a",
        ]
    elif phoneme_tokenizer.g2p_type == "pyopenjtalk_kana":
        input = "昔は俺も若かった"
        output = ["ム", "カ", "シ", "ワ", "オ", "レ", "モ", "ワ", "カ", "カ", "ッ", "タ"]
    else:
        raise NotImplementedError
    assert phoneme_tokenizer.text2tokens(input) == output


def test_token2text(phoneme_tokenizer: PhonemeTokenizer):
    assert phoneme_tokenizer.tokens2text(["a", "b", "c"]) == "abc"
