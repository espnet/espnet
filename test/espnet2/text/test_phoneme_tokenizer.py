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
    params.extend(["espeak_ng_english_us_word_sep"])
    params.extend(["espeak_ng_english_us_vits"])
    params.extend(["espeak_ng_hindi"])
    params.extend(["espeak_ng_italian"])
    params.extend(["espeak_ng_polish"])
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
try:
    import montreal_forced_aligner

    mfa_options = (
        "bulgarian_mfa croatian_mfa czech_mfa english_india_mfa "
        "english_nigeria_mfa english_uk_mfa english_us_arpa english_us_mfa "
        "french_mfa german_mfa hausa_mfa japanese_mfa korean_jamo_mfa "
        "korean_mfa polish_mfa portuguese_brazil_mfa portuguese_portugal_mfa "
        "russian_mfa spanish_latin_america_mfa spanish_spain_mfa swahili_mfa "
        "swedish_mfa "
        # "tamil_mfa "
        "thai_mfa turkish_mfa ukrainian_mfa "
        "vietnamese_hanoi_mfa vietnamese_ho_chi_minh_city_mfa "
        "vietnamese_hue_mfa"
    )
    for mfa_option in mfa_options.split():
        g2p_choice = "mfa_" + mfa_option
        params.append(g2p_choice)
        params.append(g2p_choice + "_no_space")
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
        output = [
            "ム",
            "カ",
            "シ",
            "ワ",
            "、",
            "オ",
            "レ",
            "モ",
            "ワ",
            "カ",
            "カ",
            "ッ",
            "タ",
        ]
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
            "uen1",
            "uan2",
            "h",
            "ua2",
            "t",
            "i1",
            "。",
        ]
    elif phoneme_tokenizer.g2p_type == "espeak_ng_arabic":
        input = "السلام عليكم"
        output = ["ʔ", "a", "s", "s", "ˈa", "l", "aː", "m", "ʕ", "l", "ˈiː", "k", "m"]
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
    elif phoneme_tokenizer.g2p_type == "espeak_ng_italian":
        input = "Ciao mondo."
        output = ["tʃ", "ˈa", "o", "m", "ˈo", "n", "d", "o", "."]
    elif phoneme_tokenizer.g2p_type == "espeak_ng_polish":
        input = "Witaj świecie."
        output = ["v", "ˈi", "t", "a", "j", "ɕ", "fʲ", "ˈɛ", "tɕ", "ɛ", "."]
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
    elif phoneme_tokenizer.g2p_type == "espeak_ng_english_us_word_sep":
        input = "NGINX?? mysql!!in the... way"
        output = [
            "ˈɛ",
            "n",
            "dʒ",
            "ɪ",
            "n",
            "|",
            "ˌɛ",
            "k",
            "s",
            "|",
            "??",
            "m",
            "ˌaɪ",
            "|",
            "ɛ",
            "s",
            "k",
            "j",
            "uː",
            "ˈɛ",
            "l",
            "|",
            "!!ɪ",
            "n",
            "ð",
            "ə",
            "|",
            "...",
            "w",
            "ˈeɪ",
            "|",
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
    elif phoneme_tokenizer.g2p_type == "mfa_bulgarian_mfa":
        input = "Здравей свят"
        output = ["z̪", "d̪", "r", "a", "v", "ɛ", "j", " ", "s̪", "vʲ", "a", "t̪"]
    elif phoneme_tokenizer.g2p_type == "mfa_bulgarian_mfa_no_space":
        input = "Здравей свят"
        output = ["z̪", "d̪", "r", "a", "v", "ɛ", "j", "s̪", "vʲ", "a", "t̪"]

    elif phoneme_tokenizer.g2p_type == "mfa_croatian_mfa":
        input = "Pozdrav svijete"
        output = [
            "p",
            "o˦˨",
            "z̪",
            "d̪",
            "r",
            "aː",
            "ʋ",
            " ",
            "s̪",
            "ʋ",
            "j",
            "eː˦˨",
            "t̪",
            "e",
        ]

    elif phoneme_tokenizer.g2p_type == "mfa_croatian_mfa_no_space":
        input = "Pozdrav svijete"
        output = ["p", "o˦˨", "z̪", "d̪", "r", "aː", "ʋ", "s̪", "ʋ", "j", "eː˦˨", "t̪", "e"]
    elif phoneme_tokenizer.g2p_type == "mfa_czech_mfa":
        input = "Ahoj světe"
        output = ["a", "ɦ", "o", "j", " ", "s", "v", "j", "ɛ", "t", "ɛ"]

    elif phoneme_tokenizer.g2p_type == "mfa_czech_mfa_no_space":
        input = "Ahoj světe"
        output = ["a", "ɦ", "o", "j", "s", "v", "j", "ɛ", "t", "ɛ"]

    elif phoneme_tokenizer.g2p_type == "mfa_english_india_mfa":
        input = "Hello world"
        output = ["h", "ɛ", "l", "oː", " ", "ʋ", "ɜː", "l", "ɖ"]

    elif phoneme_tokenizer.g2p_type == "mfa_english_india_mfa_no_space":
        input = "Hello world"
        output = ["h", "ɛ", "l", "oː", "ʋ", "ɜː", "l", "ɖ"]

    elif phoneme_tokenizer.g2p_type == "mfa_english_nigeria_mfa":
        input = "Hello world"
        output = ["h", "ɛ", "l", "o", " ", "w", "ɔ", "ɫ", "d"]

    elif phoneme_tokenizer.g2p_type == "mfa_english_nigeria_mfa_no_space":
        input = "Hello world"
        output = ["h", "ɛ", "l", "o", "w", "ɔ", "ɫ", "d"]

    elif phoneme_tokenizer.g2p_type == "mfa_english_uk_mfa":
        input = "Hello world"
        output = ["h", "ɛ", "l", "əw", " ", "w", "ɜː", "ɫ", "d"]

    elif phoneme_tokenizer.g2p_type == "mfa_english_uk_mfa_no_space":
        input = "Hello world"
        output = ["h", "ɛ", "l", "əw", "w", "ɜː", "ɫ", "d"]

    elif phoneme_tokenizer.g2p_type == "mfa_english_us_arpa":
        input = "Hello world"
        output = ["HH", "EH1", "L", "OW0", " ", "W", "ER1", "L", "D"]

    elif phoneme_tokenizer.g2p_type == "mfa_english_us_arpa_no_space":
        input = "Hello world"
        output = ["HH", "EH1", "L", "OW0", "W", "ER1", "L", "D"]

    elif phoneme_tokenizer.g2p_type == "mfa_english_us_mfa":
        input = "Hello world"
        output = ["h", "ɛ", "l", "ow", " ", "w", "ɝ", "ɫ", "d"]

    elif phoneme_tokenizer.g2p_type == "mfa_english_us_mfa_no_space":
        input = "Hello world"
        output = ["h", "ɛ", "l", "ow", "w", "ɝ", "ɫ", "d"]

    elif phoneme_tokenizer.g2p_type == "mfa_french_mfa":
        input = "Bonjour le monde"
        output = ["b", "ɔ̃", "ʒ", "u", "ʁ", " ", "l", " ", "m", "ɔ̃", "d"]

    elif phoneme_tokenizer.g2p_type == "mfa_french_mfa_no_space":
        input = "Bonjour le monde"
        output = ["b", "ɔ̃", "ʒ", "u", "ʁ", "l", "m", "ɔ̃", "d"]

    elif phoneme_tokenizer.g2p_type == "mfa_german_mfa":
        input = "Hallo Welt"
        output = ["h", "a", "l", "ɔ", " ", "v", "ɛ", "l", "t"]

    elif phoneme_tokenizer.g2p_type == "mfa_german_mfa_no_space":
        input = "Hallo Welt"
        output = ["h", "a", "l", "ɔ", "v", "ɛ", "l", "t"]

    elif phoneme_tokenizer.g2p_type == "mfa_hausa_mfa":
        input = "Sannu Duniya"
        output = ["s", "a˥", "n", "ɪ˩", " ", "d", "uː˥", "n", "ɪ˥", "j", "ɛ˩"]

    elif phoneme_tokenizer.g2p_type == "mfa_hausa_mfa_no_space":
        input = "Sannu Duniya"
        output = ["s", "a˥", "n", "ɪ˩", "d", "uː˥", "n", "ɪ˥", "j", "ɛ˩"]

    elif phoneme_tokenizer.g2p_type == "mfa_japanese_mfa":
        input = "昔は、俺も若かった"
        output = [
            "m",
            "ɯ",
            "h",
            "a",
            "、",
            "o",
            "ɾ",
            "m",
            "o",
            "w",
            "a",
            "k",
            "a",
            "tː",
            "a",
        ]

    elif phoneme_tokenizer.g2p_type == "mfa_japanese_mfa_no_space":
        input = "昔は、俺も若かった"
        output = [
            "m",
            "ɯ",
            "h",
            "a",
            "、",
            "o",
            "ɾ",
            "m",
            "o",
            "w",
            "a",
            "k",
            "a",
            "tː",
            "a",
        ]

    elif phoneme_tokenizer.g2p_type == "mfa_korean_jamo_mfa":
        input = "ㄴㅏㄴㅡㄴ ㅎㅏㄱㄱㅛㅇㅔ ㄱㅏㅂㄴㅣㄷㅏ."
        output = [
            "n",
            "ɐ",
            "n",
            "ɨ",
            "n",
            " ",
            "h",
            "ɐ",
            "k̚",
            "kʰ",
            "j",
            "o",
            "e",
            " ",
            "k",
            "ɐ",
            "m",
            "n",
            "i",
            "d",
            "ɐ",
            ".",
        ]

    elif phoneme_tokenizer.g2p_type == "mfa_korean_jamo_mfa_no_space":
        input = "ㄴㅏㄴㅡㄴ ㅎㅏㄱㄱㅛㅇㅔ ㄱㅏㅂㄴㅣㄷㅏ."
        output = [
            "n",
            "ɐ",
            "n",
            "ɨ",
            "n",
            "h",
            "ɐ",
            "k̚",
            "kʰ",
            "j",
            "o",
            "e",
            "k",
            "ɐ",
            "m",
            "n",
            "i",
            "d",
            "ɐ",
            ".",
        ]

    elif phoneme_tokenizer.g2p_type == "mfa_korean_mfa":
        input = "나는 학교에 갑니다."
        output = [
            "n",
            "ɐ",
            "n",
            "ɨ",
            "n",
            " ",
            "h",
            "ɐ",
            "k̚",
            "kʰ",
            "j",
            "o",
            "e",
            " ",
            "k",
            "ɐ",
            "m",
            "n",
            "i",
            "d",
            "ɐ",
            ".",
        ]

    elif phoneme_tokenizer.g2p_type == "mfa_korean_mfa_no_space":
        input = "나는 학교에 갑니다."
        output = [
            "n",
            "ɐ",
            "n",
            "ɨ",
            "n",
            "h",
            "ɐ",
            "k̚",
            "kʰ",
            "j",
            "o",
            "e",
            "k",
            "ɐ",
            "m",
            "n",
            "i",
            "d",
            "ɐ",
            ".",
        ]

    elif phoneme_tokenizer.g2p_type == "mfa_polish_mfa":
        input = "Witaj świecie"
        output = ["vʲ", "i", "t̪", "a", "j", " ", "ɕ", "fʲ", "ɛ", "tɕ", "ɛ"]

    elif phoneme_tokenizer.g2p_type == "mfa_polish_mfa_no_space":
        input = "Witaj świecie"
        output = ["vʲ", "i", "t̪", "a", "j", "ɕ", "fʲ", "ɛ", "tɕ", "ɛ"]

    elif phoneme_tokenizer.g2p_type == "mfa_portuguese_brazil_mfa":
        input = "Olá Mundo"
        output = ["o", "l", "a", " ", "m", "ũ", "d", "u"]

    elif phoneme_tokenizer.g2p_type == "mfa_portuguese_brazil_mfa_no_space":
        input = "Olá Mundo"
        output = ["o", "l", "a", "m", "ũ", "d", "u"]

    elif phoneme_tokenizer.g2p_type == "mfa_portuguese_portugal_mfa":
        input = "Olá Mundo"
        output = ["ɔ", "l", "a", " ", "m", "ũ", "d", "u"]

    elif phoneme_tokenizer.g2p_type == "mfa_portuguese_portugal_mfa_no_space":
        input = "Olá Mundo"
        output = ["ɔ", "l", "a", "m", "ũ", "d", "u"]

    elif phoneme_tokenizer.g2p_type == "mfa_russian_mfa":
        input = "Привет, мир"
        output = ["p", "rʲ", "ɪ", "vʲ", "e", "t̪", ",", " ", "mʲ", "i", "r"]

    elif phoneme_tokenizer.g2p_type == "mfa_russian_mfa_no_space":
        input = "Привет, мир"
        output = ["p", "rʲ", "ɪ", "vʲ", "e", "t̪", ",", "mʲ", "i", "r"]

    elif phoneme_tokenizer.g2p_type == "mfa_spanish_latin_america_mfa":
        input = "Hola Mundo"
        output = ["o", "l", "a", " ", "m", "u", "n", "d̪", "o"]

    elif phoneme_tokenizer.g2p_type == "mfa_spanish_latin_america_mfa_no_space":
        input = "Hola Mundo"
        output = ["o", "l", "a", "m", "u", "n", "d̪", "o"]

    elif phoneme_tokenizer.g2p_type == "mfa_spanish_spain_mfa":
        input = "Hola Mundo"
        output = ["o", "l", "a", " ", "m", "u", "n", "d̪", "o"]

    elif phoneme_tokenizer.g2p_type == "mfa_spanish_spain_mfa_no_space":
        input = "Hola Mundo"
        output = ["o", "l", "a", "m", "u", "n", "d̪", "o"]

    elif phoneme_tokenizer.g2p_type == "mfa_swahili_mfa":
        input = "Salamu, Dunia"
        output = ["s", "ɑ", "l", "ɑ", "m", "u", ",", " ", "d", "u", "n", "i", "ɑ"]

    elif phoneme_tokenizer.g2p_type == "mfa_swahili_mfa_no_space":
        input = "Salamu, Dunia"
        output = ["s", "ɑ", "l", "ɑ", "m", "u", ",", "d", "u", "n", "i", "ɑ"]

    elif phoneme_tokenizer.g2p_type == "mfa_swedish_mfa":
        input = "Hej världen"
        output = ["h", "ɛ", "j", " ", "ʋ", "ɛː", "ɖ", "ɛ", "n̪"]

    elif phoneme_tokenizer.g2p_type == "mfa_swedish_mfa_no_space":
        input = "Hej världen"
        output = ["h", "ɛ", "j", "ʋ", "ɛː", "ɖ", "ɛ", "n̪"]

    elif phoneme_tokenizer.g2p_type == "mfa_thai_mfa":
        input = "สวัสดีชาวโลก"
        output = [
            "s",
            "a˨˩",
            "w",
            "a˨˩",
            "t̚",
            "d",
            "iː˧",
            "tɕʰ",
            "aː˧",
            "w",
            "a˦˥",
            "l",
            "oː˥˩",
            "k̚",
        ]

    elif phoneme_tokenizer.g2p_type == "mfa_thai_mfa_no_space":
        input = "สวัสดีชาวโลก"
        output = [
            "s",
            "a˨˩",
            "w",
            "a˨˩",
            "t̚",
            "d",
            "iː˧",
            "tɕʰ",
            "aː˧",
            "w",
            "a˦˥",
            "l",
            "oː˥˩",
            "k̚",
        ]

    elif phoneme_tokenizer.g2p_type == "mfa_turkish_mfa":
        input = "Selam Dünya"
        output = ["s̪", "e", "ɫ", "a", "m", " ", "d̪", "y", "n̪", "j", "a"]

    elif phoneme_tokenizer.g2p_type == "mfa_turkish_mfa_no_space":
        input = "Selam Dünya"
        output = ["s̪", "e", "ɫ", "a", "m", "d̪", "y", "n̪", "j", "a"]

    elif phoneme_tokenizer.g2p_type == "mfa_ukrainian_mfa":
        input = "Привіт Світ"
        output = ["p", "ɾ", "e", "ʋʲ", "i", "t̪", " ", "sʲ", "ʋʲ", "i", "t̪"]

    elif phoneme_tokenizer.g2p_type == "mfa_ukrainian_mfa_no_space":
        input = "Привіт Світ"
        output = ["p", "ɾ", "e", "ʋʲ", "i", "t̪", "sʲ", "ʋʲ", "i", "t̪"]

    elif phoneme_tokenizer.g2p_type == "mfa_vietnamese_hanoi_mfa":
        input = "Chào thế giới"
        output = ["tɕ", "aː˨˨", "w", " ", "tʰ", "eː˨˦", " ", "z", "əː˨˦", "j"]

    elif phoneme_tokenizer.g2p_type == "mfa_vietnamese_hanoi_mfa_no_space":
        input = "Chào thế giới"
        output = ["tɕ", "aː˨˨", "w", "tʰ", "eː˨˦", "z", "əː˨˦", "j"]

    elif phoneme_tokenizer.g2p_type == "mfa_vietnamese_ho_chi_minh_city_mfa":
        input = "Chào thế giới"
        output = ["c", "aː˨˩", "w", " ", "tʰ", "eː˦˥", " ", "j", "əː˦˥", "j"]

    elif (
        phoneme_tokenizer.g2p_type == "mfa_vietnamese_" "ho_chi_minh_city_mfa_no_space"
    ):
        input = "Chào thế giới"
        output = ["c", "aː˨˩", "w", "tʰ", "eː˦˥", "j", "əː˦˥", "j"]

    elif phoneme_tokenizer.g2p_type == "mfa_vietnamese_hue_mfa":
        input = "Chào thế giới"
        output = ["c", "aː˦˨", "w", " ", "tʰ", "eː˨ˀ˦", " ", "j", "əː˨ˀ˦", "j"]

    elif phoneme_tokenizer.g2p_type == "mfa_vietnamese_hue_mfa_no_space":
        input = "Chào thế giới"
        output = ["c", "aː˦˨", "w", "tʰ", "eː˨ˀ˦", "j", "əː˨ˀ˦", "j"]

    elif phoneme_tokenizer.g2p_type == "is_g2p":
        input = "hlaupa í burtu í dag"
        output = [
            "l_0",
            "9i:",
            ".",
            "p",
            "a",
            ",",
            "i:",
            ",",
            "p",
            "Y",
            "r_0",
            ".",
            "t",
            "Y",
            ",",
            "i:",
            ",",
            "t",
            "a:",
            "G",
        ]
    else:
        raise NotImplementedError
    assert phoneme_tokenizer.text2tokens(input) == output


def test_token2text(phoneme_tokenizer: PhonemeTokenizer):
    assert phoneme_tokenizer.tokens2text(["a", "b", "c"]) == "abc"
