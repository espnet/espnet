import pytest

from espnet2.text.phoneme_tokenizer import PhonemeTokenizer

params = [None, "g2p_en", "g2p_en_no_space"]
try:
    import pyopenjtalk

    params.extend(
        [
            "pyopenjtalk",
            "pyopenjtalk_accent",
            "pyopenjtalk_kana",
            "pyopenjtalk_accent_with_pause",
        ]
    )
    del pyopenjtalk
except ImportError:
    pass
try:
    import pypinyin

    params.extend(["pypinyin_g2p", "pypinyin_g2p_phone"])
    del pypinyin
except ImportError:
    pass
try:
    import phonemizer

    params.extend(["espeak_ng_arabic"])
    del phonemizer
except ImportError:
    pass


@pytest.fixture(params=params)
def phoneme_tokenizer(request):
    return PhonemeTokenizer(g2p_type=request.param)


def test_repr(phoneme_tokenizer: PhonemeTokenizer):
    print(phoneme_tokenizer)


@pytest.mark.execution_timeout(5)
def test_text2tokens(phoneme_tokenizer: PhonemeTokenizer):
    if phoneme_tokenizer.g2p_type is None:
        input = "HH AH0 L OW1 <space> W ER1 L D"
        output = ["HH", "AH0", "L", "OW1", "<space>", "W", "ER1", "L", "D"]
    elif phoneme_tokenizer.g2p_type == "g2p_en":
        input = "Hello World"
        output = ["HH", "AH0", "L", "OW1", " ", "W", "ER1", "L", "D"]
    elif phoneme_tokenizer.g2p_type == "g2p_en_no_space":
        input = "Hello World"
        output = ["HH", "AH0", "L", "OW1", "W", "ER1", "L", "D"]
    elif phoneme_tokenizer.g2p_type == "pyopenjtalk":
        input = "昔は、俺も若かった"
        output = [
            "m",
            "u",
            "k",
            "a",
            "sh",
            "i",
            "w",
            "a",
            "pau",
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
        input = "昔は、俺も若かった"
        output = ["ム", "カ", "シ", "ワ", "、", "オ", "レ", "モ", "ワ", "カ", "カ", "ッ", "タ"]
    elif phoneme_tokenizer.g2p_type == "pyopenjtalk_accent":
        input = "昔は、俺も若かった"
        output = [
            "m",
            "4",
            "-3",
            "u",
            "4",
            "-3",
            "k",
            "4",
            "-2",
            "a",
            "4",
            "-2",
            "sh",
            "4",
            "-1",
            "i",
            "4",
            "-1",
            "w",
            "4",
            "0",
            "a",
            "4",
            "0",
            "o",
            "3",
            "-2",
            "r",
            "3",
            "-1",
            "e",
            "3",
            "-1",
            "m",
            "3",
            "0",
            "o",
            "3",
            "0",
            "w",
            "2",
            "-1",
            "a",
            "2",
            "-1",
            "k",
            "2",
            "0",
            "a",
            "2",
            "0",
            "k",
            "2",
            "1",
            "a",
            "2",
            "1",
            "cl",
            "2",
            "2",
            "t",
            "2",
            "3",
            "a",
            "2",
            "3",
        ]
    elif phoneme_tokenizer.g2p_type == "pyopenjtalk_accent_with_pause":
        input = "昔は、俺も若かった"
        output = [
            "m",
            "4",
            "-3",
            "u",
            "4",
            "-3",
            "k",
            "4",
            "-2",
            "a",
            "4",
            "-2",
            "sh",
            "4",
            "-1",
            "i",
            "4",
            "-1",
            "w",
            "4",
            "0",
            "a",
            "4",
            "0",
            "pau",
            "o",
            "3",
            "-2",
            "r",
            "3",
            "-1",
            "e",
            "3",
            "-1",
            "m",
            "3",
            "0",
            "o",
            "3",
            "0",
            "w",
            "2",
            "-1",
            "a",
            "2",
            "-1",
            "k",
            "2",
            "0",
            "a",
            "2",
            "0",
            "k",
            "2",
            "1",
            "a",
            "2",
            "1",
            "cl",
            "2",
            "2",
            "t",
            "2",
            "3",
            "a",
            "2",
            "3",
        ]
    elif phoneme_tokenizer.g2p_type == "pypinyin_g2p":
        input = "卡尔普陪外孙玩滑梯。"
        output = [
            "ka3",
            "er3",
            "pu3",
            "pei2",
            "wai4",
            "sun1",
            "wan2",
            "hua2",
            "ti1",
            "。",
        ]
    elif phoneme_tokenizer.g2p_type == "pypinyin_g2p_phone":
        input = "卡尔普陪外孙玩滑梯。"
        output = [
            "k",
            "a3",
            "er3",
            "p",
            "u3",
            "p",
            "ei2",
            "uai4",
            "s",
            "un1",
            "uan2",
            "h",
            "ua2",
            "t",
            "i1",
            "。",
        ]
    elif phoneme_tokenizer.g2p_type == "espeak_ng_arabic":
        input = u"السلام عليكم"
        output = ["ʔ", "a", "s", "s", "ˈa", "l", "aː", "m", "ʕ", "l", "ˈiː", "k", "m"]
    else:
        raise NotImplementedError
    assert phoneme_tokenizer.text2tokens(input) == output


def test_token2text(phoneme_tokenizer: PhonemeTokenizer):
    assert phoneme_tokenizer.tokens2text(["a", "b", "c"]) == "abc"
