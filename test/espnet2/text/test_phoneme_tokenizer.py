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
            "pyopenjtalk_prosody",
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
    params.extend(["espeak_ng_german"])
    params.extend(["espeak_ng_french"])
    params.extend(["espeak_ng_spanish"])
    params.extend(["espeak_ng_russian"])
    params.extend(["espeak_ng_greek"])
    params.extend(["espeak_ng_finnish"])
    params.extend(["espeak_ng_hungarian"])
    params.extend(["espeak_ng_dutch"])
    params.extend(["espeak_ng_english_us_vits"])
    params.extend(["espeak_ng_hindi"])
    del phonemizer
except ImportError:
    pass
try:
    import g2pk

    params.extend(["g2pk", "g2pk_no_space"])
    del g2pk
except ImportError:
    pass
params.extend(["korean_jaso", "korean_jaso_no_space"])


@pytest.fixture(params=params)
def phoneme_tokenizer(request):
    return PhonemeTokenizer(g2p_type=request.param)


def test_repr(phoneme_tokenizer: PhonemeTokenizer):
    print(phoneme_tokenizer)


@pytest.mark.execution_timeout(5)
def test_text2tokens(phoneme_tokenizer: PhonemeTokenizer):
    if phoneme_tokenizer.g2p_type is None:
        input = "HH AH0 L OW1   W ER1 L D"
        output = ["HH", "AH0", "L", "OW1", " ", "W", "ER1", "L", "D"]
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
    elif phoneme_tokenizer.g2p_type == "pyopenjtalk_prosody":
        input = "昔は、俺も若かった"
        output = [
            "^",
            "m",
            "u",
            "[",
            "k",
            "a",
            "sh",
            "i",
            "w",
            "a",
            "_",
            "o",
            "[",
            "r",
            "e",
            "m",
            "o",
            "#",
            "w",
            "a",
            "[",
            "k",
            "a",
            "]",
            "k",
            "a",
            "cl",
            "t",
            "a",
            "$",
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
        input = "السلام عليكم"
        output = ["ʔ", "a", "s", "s", "a", "l", "ˈaː", "m", "ʕ", "l", "ˈiː", "k", "m"]
    elif phoneme_tokenizer.g2p_type == "espeak_ng_german":
        input = "Das hört sich gut an."
        output = [
            "d",
            "a",
            "s",
            "h",
            "ˈœ",
            "ɾ",
            "t",
            "z",
            "ɪ",
            "ç",
            "ɡ",
            "ˈuː",
            "t",
            "ˈa",
            "n",
            ".",
        ]
    elif phoneme_tokenizer.g2p_type == "espeak_ng_french":
        input = "Bonjour le monde."
        output = ["b", "ɔ̃", "ʒ", "ˈu", "ʁ", "l", "ə-", "m", "ˈɔ̃", "d", "."]
    elif phoneme_tokenizer.g2p_type == "espeak_ng_spanish":
        input = "Hola Mundo."
        output = ["ˈo", "l", "a", "m", "ˈu", "n", "d", "o", "."]
    elif phoneme_tokenizer.g2p_type == "espeak_ng_russian":
        input = "Привет мир."
        output = ["p", "rʲ", "i", "vʲ", "ˈe", "t", "mʲ", "ˈi", "r", "."]
    elif phoneme_tokenizer.g2p_type == "espeak_ng_greek":
        input = "Γειά σου Κόσμε."
        output = ["j", "ˈa", "s", "u", "k", "ˈo", "s", "m", "e", "."]
    elif phoneme_tokenizer.g2p_type == "espeak_ng_finnish":
        input = "Hei maailma."
        output = ["h", "ˈei", "m", "ˈaː", "ɪ", "l", "m", "a", "."]
    elif phoneme_tokenizer.g2p_type == "espeak_ng_hungarian":
        input = "Helló Világ."
        output = ["h", "ˈɛ", "l", "l", "oː", "v", "ˈi", "l", "aː", "ɡ", "."]
    elif phoneme_tokenizer.g2p_type == "espeak_ng_dutch":
        input = "Hallo Wereld."
        output = ["h", "ˈɑ", "l", "oː", "ʋ", "ˈɪː", "r", "ə", "l", "t", "."]
    elif phoneme_tokenizer.g2p_type == "espeak_ng_hindi":
        input = "नमस्ते दुनिया"
        output = ["n", "ə", "m", "ˈʌ", "s", "t", "eː", "d", "ˈʊ", "n", "ɪ", "j", "ˌaː"]
    elif phoneme_tokenizer.g2p_type == "g2pk":
        input = "안녕하세요 세계입니다."
        output = [
            "ᄋ",
            "ᅡ",
            "ᆫ",
            "ᄂ",
            "ᅧ",
            "ᆼ",
            "ᄒ",
            "ᅡ",
            "ᄉ",
            "ᅦ",
            "ᄋ",
            "ᅭ",
            " ",
            "ᄉ",
            "ᅦ",
            "ᄀ",
            "ᅨ",
            "ᄋ",
            "ᅵ",
            "ᆷ",
            "ᄂ",
            "ᅵ",
            "ᄃ",
            "ᅡ",
            ".",
        ]
    elif phoneme_tokenizer.g2p_type == "g2pk_no_space":
        input = "안녕하세요 세계입니다."
        output = [
            "ᄋ",
            "ᅡ",
            "ᆫ",
            "ᄂ",
            "ᅧ",
            "ᆼ",
            "ᄒ",
            "ᅡ",
            "ᄉ",
            "ᅦ",
            "ᄋ",
            "ᅭ",
            "ᄉ",
            "ᅦ",
            "ᄀ",
            "ᅨ",
            "ᄋ",
            "ᅵ",
            "ᆷ",
            "ᄂ",
            "ᅵ",
            "ᄃ",
            "ᅡ",
            ".",
        ]
    elif phoneme_tokenizer.g2p_type == "espeak_ng_english_us_vits":
        input = "Hello, World."
        output = [
            "h",
            "ə",
            "l",
            "ˈ",
            "o",
            "ʊ",
            ",",
            "<space>",
            "w",
            "ˈ",
            "ɜ",
            "ː",
            "l",
            "d",
            ".",
        ]
    elif phoneme_tokenizer.g2p_type == "korean_jaso":
        input = "나는 학교에 갑니다."
        output = [
            "ᄂ",
            "ᅡ",
            "ᄂ",
            "ᅳ",
            "ᆫ",
            "<space>",
            "ᄒ",
            "ᅡ",
            "ᆨ",
            "ᄀ",
            "ᅭ",
            "ᄋ",
            "ᅦ",
            "<space>",
            "ᄀ",
            "ᅡ",
            "ᆸ",
            "ᄂ",
            "ᅵ",
            "ᄃ",
            "ᅡ",
            ".",
        ]
    elif phoneme_tokenizer.g2p_type == "korean_jaso_no_space":
        input = "나는 학교에 갑니다."
        output = [
            "ᄂ",
            "ᅡ",
            "ᄂ",
            "ᅳ",
            "ᆫ",
            "ᄒ",
            "ᅡ",
            "ᆨ",
            "ᄀ",
            "ᅭ",
            "ᄋ",
            "ᅦ",
            "ᄀ",
            "ᅡ",
            "ᆸ",
            "ᄂ",
            "ᅵ",
            "ᄃ",
            "ᅡ",
            ".",
        ]
    else:
        raise NotImplementedError
    assert phoneme_tokenizer.text2tokens(input) == output


def test_token2text(phoneme_tokenizer: PhonemeTokenizer):
    assert phoneme_tokenizer.tokens2text(["a", "b", "c"]) == "abc"
