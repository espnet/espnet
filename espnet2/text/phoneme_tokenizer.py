import logging
import re
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Union

import g2p_en
import jamo
from packaging.version import parse as V
from typeguard import typechecked

from espnet2.text.abs_tokenizer import AbsTokenizer

g2p_choices = [
    None,
    "g2p_en",
    "g2p_en_no_space",
    "pyopenjtalk",
    "pyopenjtalk_kana",
    "pyopenjtalk_accent",
    "pyopenjtalk_accent_with_pause",
    "pyopenjtalk_prosody",
    "pypinyin_g2p",
    "pypinyin_g2p_phone",
    "pypinyin_g2p_phone_without_prosody",
    "espeak_ng_arabic",
    "espeak_ng_german",
    "espeak_ng_french",
    "espeak_ng_spanish",
    "espeak_ng_russian",
    "espeak_ng_greek",
    "espeak_ng_finnish",
    "espeak_ng_hungarian",
    "espeak_ng_dutch",
    "espeak_ng_english_us_vits",
    "espeak_ng_hindi",
    "espeak_ng_italian",
    "espeak_ng_ukrainian",
    "espeak_ng_polish",
    "g2pk",
    "g2pk_no_space",
    "g2pk_explicit_space",
    "korean_jaso",
    "korean_jaso_no_space",
    "g2p_is",
]


def split_by_space(text) -> List[str]:
    """
    Splits the input text into a list of words based on spaces.

    This function replaces multiple consecutive spaces with a single space,
    ensuring that the output list contains words separated by a single space.

    Args:
        text (str): The input text string to be split.

    Returns:
        List[str]: A list of words extracted from the input text.

    Examples:
        >>> split_by_space("Hello world")
        ['Hello', 'world']

        >>> split_by_space("This  is  a   test")
        ['This', 'is', 'a', 'test']

        >>> split_by_space("  Leading and trailing spaces  ")
        ['Leading', 'and', 'trailing', 'spaces']

        >>> split_by_space("Multiple   spaces  here")
        ['Multiple', 'spaces', 'here']
    """
    if "   " in text:
        text = text.replace("   ", " <space> ")
        return [c.replace("<space>", " ") for c in text.split(" ")]
    else:
        return text.split(" ")


def pyopenjtalk_g2p(text) -> List[str]:
    """
        Converts input text to phonemes using the pyopenjtalk library.

    This function utilizes the `pyopenjtalk` library to perform
    grapheme-to-phoneme (G2P) conversion, generating a list of
    phonemes represented as strings.

    Args:
        text (str): The input text that needs to be converted to phonemes.

    Returns:
        List[str]: A list of phonemes extracted from the input text.

    Examples:
        >>> phonemes = pyopenjtalk_g2p("こんにちは。")
        >>> print(phonemes)
        ['k', 'o', 'N', 'n', 'i', 'ch', 'i', 'w', 'a']

    Note:
        The input text should be in Japanese for accurate phoneme
        extraction. Ensure that the `pyopenjtalk` library is installed
        and available in your environment.
    """
    import pyopenjtalk

    # phones is a str object separated by space
    phones = pyopenjtalk.g2p(text, kana=False)
    phones = phones.split(" ")
    return phones


def _extract_fullcontext_label(text):
    import pyopenjtalk

    if V(pyopenjtalk.__version__) >= V("0.3.0"):
        return pyopenjtalk.make_label(pyopenjtalk.run_frontend(text))
    else:
        return pyopenjtalk.run_frontend(text)[1]


def pyopenjtalk_g2p_accent(text) -> List[str]:
    """
    Convert input text to phonemes with accent information.

    This function uses the PyOpenJTalk library to extract phonemes from
    input text while also incorporating accentuation details. It returns
    a list of phonemes, each represented as a string, where the accent
    type and position are included.

    Args:
        text (str): The input text to be converted into phonemes.

    Returns:
        List[str]: A list of phonemes with accent information.

    Examples:
        >>> pyopenjtalk_g2p_accent("こんにちは。")
        ['k', 'o', 'N', 'n', 'i', 'ch', 'i', 'w', 'a']

    Note:
        This function assumes that the input text is in Japanese.
    """
    phones = []
    for labels in _extract_fullcontext_label(text):
        p = re.findall(r"\-(.*?)\+.*?\/A:([0-9\-]+).*?\/F:.*?_([0-9]+)", labels)
        if len(p) == 1:
            phones += [p[0][0], p[0][2], p[0][1]]
    return phones


def pyopenjtalk_g2p_accent_with_pause(text) -> List[str]:
    """
        Convert text to a sequence of phonemes with accent and pause information.

    This function processes the input text to extract phonemes while
    considering accent information and pauses. It identifies pauses in
    the input and represents them as 'pau' in the output list. The
    function utilizes full-context labels extracted from the input text
    to determine phoneme attributes.

    Args:
        text (str): Input text to be converted into phonemes.

    Returns:
        List[str]: A list of phonemes including accents and pauses.

    Examples:
        >>> result = pyopenjtalk_g2p_accent_with_pause("こんにちは")
        >>> print(result)
        ['k', 'o', 'N', 'n', 'i', 'ch', 'i']

        >>> result_with_pause = pyopenjtalk_g2p_accent_with_pause("こんにちは。")
        >>> print(result_with_pause)
        ['k', 'o', 'N', 'n', 'i', 'ch', 'i', 'pau']

    Note:
        The function relies on the `_extract_fullcontext_label`
        function to get full-context labels from the input text.
        It requires the `pyopenjtalk` package to be installed.
    """
    phones = []
    for labels in _extract_fullcontext_label(text):
        if labels.split("-")[1].split("+")[0] == "pau":
            phones += ["pau"]
            continue
        p = re.findall(r"\-(.*?)\+.*?\/A:([0-9\-]+).*?\/F:.*?_([0-9]+)", labels)
        if len(p) == 1:
            phones += [p[0][0], p[0][2], p[0][1]]
    return phones


def pyopenjtalk_g2p_kana(text) -> List[str]:
    """
        Converts input text to its corresponding kana representation using
    PyOpenJTalk.

    This function utilizes the PyOpenJTalk library to perform grapheme-to-phoneme
    conversion specifically for kana. The input text is transformed into its
    phonetic representation in kana format.

    Args:
        text (str): The input text to be converted into kana.

    Returns:
        List[str]: A list of kana characters corresponding to the input text.

    Examples:
        >>> kana_output = pyopenjtalk_g2p_kana("こんにちは")
        >>> print(kana_output)
        ['こ', 'ん', 'に', 'ち', 'は']

    Note:
        Ensure that the PyOpenJTalk library is installed and properly configured
        in your environment to use this function.
    """
    import pyopenjtalk

    kanas = pyopenjtalk.g2p(text, kana=True)
    return list(kanas)


def pyopenjtalk_g2p_prosody(text: str, drop_unvoiced_vowels: bool = True) -> List[str]:
    """
        Extract phoneme + prosody symbol sequence from input full-context labels.

    The algorithm is based on `Prosodic features control by symbols as input of
    sequence-to-sequence acoustic modeling for neural TTS`_ with some r9y9's tweaks.

    Args:
        text (str): Input text.
        drop_unvoiced_vowels (bool): Whether to drop unvoiced vowels.

    Returns:
        List[str]: List of phoneme + prosody symbols.

    Examples:
        >>> from espnet2.text.phoneme_tokenizer import pyopenjtalk_g2p_prosody
        >>> pyopenjtalk_g2p_prosody("こんにちは。")
        ['^', 'k', 'o', '[', 'N', 'n', 'i', 'ch', 'i', 'w', 'a', '$']

    .. _`Prosodic features control by symbols as input of sequence-to-sequence acoustic
        modeling for neural TTS`: https://doi.org/10.1587/transinf.2020EDP7104
    """
    labels = _extract_fullcontext_label(text)
    N = len(labels)

    phones = []
    for n in range(N):
        lab_curr = labels[n]

        # current phoneme
        p3 = re.search(r"\-(.*?)\+", lab_curr).group(1)

        # deal unvoiced vowels as normal vowels
        if drop_unvoiced_vowels and p3 in "AEIOU":
            p3 = p3.lower()

        # deal with sil at the beginning and the end of text
        if p3 == "sil":
            assert n == 0 or n == N - 1
            if n == 0:
                phones.append("^")
            elif n == N - 1:
                # check question form or not
                e3 = _numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                if e3 == 0:
                    phones.append("$")
                elif e3 == 1:
                    phones.append("?")
            continue
        elif p3 == "pau":
            phones.append("_")
            continue
        else:
            phones.append(p3)

        # accent type and position info (forward or backward)
        a1 = _numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
        a2 = _numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
        a3 = _numeric_feature_by_regex(r"\+(\d+)/", lab_curr)

        # number of mora in accent phrase
        f1 = _numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)

        a2_next = _numeric_feature_by_regex(r"\+(\d+)\+", labels[n + 1])
        # accent phrase border
        if a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNcl":
            phones.append("#")
        # pitch falling
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            phones.append("]")
        # pitch rising
        elif a2 == 1 and a2_next == 2:
            phones.append("[")

    return phones


def _numeric_feature_by_regex(regex, s):
    match = re.search(regex, s)
    if match is None:
        return -50
    return int(match.group(1))


def pypinyin_g2p(text) -> List[str]:
    """
        Convert Chinese text to pinyin phonemes.

    This function utilizes the `pypinyin` library to convert the given Chinese
    text into its corresponding pinyin phonemes. It extracts the pinyin with
    tone numbers, returning them as a list of strings.

    Args:
        text (str): The Chinese text to be converted into pinyin.

    Returns:
        List[str]: A list of pinyin phonemes corresponding to the input text.

    Examples:
        >>> pypinyin_g2p("你好")
        ['nǐ', 'hǎo']

        >>> pypinyin_g2p("中国")
        ['zhōng', 'guó']

    Note:
        Ensure that the `pypinyin` library is installed in your environment.
    """
    from pypinyin import Style, pinyin

    phones = [phone[0] for phone in pinyin(text, style=Style.TONE3)]
    return phones


def pypinyin_g2p_phone(text) -> List[str]:
    """
        Convert Chinese text to phonemes using Pinyin with tone markings.

    This function takes a Chinese text input and converts it into a list of phonemes
    based on the Pinyin representation, including initial and final sounds, while
    maintaining tone information. The function is useful for applications requiring
    phonetic analysis or text-to-speech systems.

    Args:
        text (str): The input Chinese text to be converted to phonemes.

    Returns:
        List[str]: A list of phonemes extracted from the input text.

    Examples:
        >>> pypinyin_g2p_phone("你好")
        ['n', 'i', 'h', 'a', 'o']

        >>> pypinyin_g2p_phone("北京")
        ['b', 'e', 'i', 'j', 'i', 'n']

    Note:
        This function utilizes the `pypinyin` library, which must be installed for
        the function to work correctly. Ensure to handle any potential exceptions
        that may arise from invalid input.
    """
    from pypinyin import Style, pinyin
    from pypinyin.style._utils import get_finals, get_initials

    phones = [
        p
        for phone in pinyin(text, style=Style.TONE3)
        for p in [
            get_initials(phone[0], strict=True),
            (
                get_finals(phone[0][:-1], strict=True) + phone[0][-1]
                if phone[0][-1].isdigit()
                else (
                    get_finals(phone[0], strict=True)
                    if phone[0][-1].isalnum()
                    else phone[0]
                )
            ),
        ]
        # Remove the case of individual tones as a phoneme
        if len(p) != 0 and not p.isdigit()
    ]
    return phones


def pypinyin_g2p_phone_without_prosody(text) -> List[str]:
    """
        Convert Chinese text to phonemes without prosody using pypinyin.

    This function takes a string of Chinese text and converts it into a list of
    phonemes. The conversion is done using the pypinyin library, and the output
    does not include any prosodic features.

    Args:
        text (str): The input Chinese text to be converted to phonemes.

    Returns:
        List[str]: A list of phonemes corresponding to the input text.

    Examples:
        >>> from espnet2.text.phoneme_tokenizer import pypinyin_g2p_phone_without_prosody
        >>> pypinyin_g2p_phone_without_prosody("你好")
        ['n', 'i', 'h', 'a', 'o']

    Note:
        This function uses the pypinyin library's normal style for phoneme
        conversion, which does not include tone markings or prosodic symbols.
    """
    from pypinyin import Style, pinyin
    from pypinyin.style._utils import get_finals, get_initials

    phones = []
    for phone in pinyin(text, style=Style.NORMAL, strict=False):
        initial = get_initials(phone[0], strict=False)
        final = get_finals(phone[0], strict=False)
        if len(initial) != 0:
            if initial in ["x", "y", "j", "q"]:
                if final == "un":
                    final = "vn"
                elif final == "uan":
                    final = "van"
                elif final == "u":
                    final = "v"
            if final == "ue":
                final = "ve"
            phones.append(initial + "_" + final)
        else:
            phones.append(final)
    return phones


class G2p_en:
    """
    On behalf of g2p_en.G2p.

    This class serves as a wrapper for the g2p_en.G2p class, which is used for
    converting English text to phonemes. Note that g2p_en.G2p is not
    picklable, meaning it cannot be serialized for use with the multiprocessing
    module. As a workaround, an instance of g2p_en.G2p is created upon the
    first call to this class.

    Attributes:
        no_space (bool): If True, spaces representing word separators will be
            removed from the output.

    Args:
        no_space (bool): Flag indicating whether to remove spaces from the
            phoneme output. Default is False.

    Returns:
        List[str]: A list of phonemes corresponding to the input text.

    Examples:
        >>> g2p = G2p_en(no_space=True)
        >>> phonemes = g2p("Hello world")
        >>> print(phonemes)
        ['h', 'ə', 'l', 'oʊ', 'w', 'ɜ', 'r', 'l', 'd']
    """

    def __init__(self, no_space: bool = False):
        self.no_space = no_space
        self.g2p = None

    def __call__(self, text) -> List[str]:
        if self.g2p is None:
            self.g2p = g2p_en.G2p()

        phones = self.g2p(text)
        if self.no_space:
            # remove space which represents word serapater
            phones = list(filter(lambda s: s != " ", phones))
        return phones


class G2pk:
    """
        On behalf of g2pk.G2p.

    g2pk.G2p isn't picklable and it can't be copied to other processes
    via the multiprocessing module. As a workaround, g2pk.G2p is
    instantiated upon calling this class.

    Attributes:
        descritive (bool): If True, produces descriptive phonemes.
        group_vowels (bool): If True, groups similar vowel sounds.
        to_syl (bool): If True, converts output to syllables.
        no_space (bool): If True, removes spaces that represent word separators.
        explicit_space (bool): If True, replaces spaces with a specified symbol.
        space_symbol (str): The symbol used to represent spaces.

    Args:
        descritive (bool): Optional; defaults to False.
        group_vowels (bool): Optional; defaults to False.
        to_syl (bool): Optional; defaults to False.
        no_space (bool): Optional; defaults to False.
        explicit_space (bool): Optional; defaults to False.
        space_symbol (str): Optional; defaults to "<space>".

    Returns:
        List[str]: A list of phonemes generated from the input text.

    Examples:
        >>> from espnet2.text.phoneme_tokenizer import G2pk
        >>> g2p = G2pk()
        >>> g2p("hello world")
        ['h', 'e', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']

        >>> g2p_no_space = G2pk(no_space=True)
        >>> g2p_no_space("hello world")
        ['h', 'e', 'l', 'o', 'w', 'o', 'r', 'l', 'd']

        >>> g2p_explicit_space = G2pk(explicit_space=True, space_symbol="_")
        >>> g2p_explicit_space("hello world")
        ['h', 'e', 'l', 'o', '_', 'w', 'o', 'r', 'l', 'd']
    """

    def __init__(
        self,
        descritive=False,
        group_vowels=False,
        to_syl=False,
        no_space=False,
        explicit_space=False,
        space_symbol="<space>",
    ):
        self.descritive = descritive
        self.group_vowels = group_vowels
        self.to_syl = to_syl
        self.no_space = no_space
        self.explicit_space = explicit_space
        self.space_symbol = space_symbol
        self.g2p = None

    def __call__(self, text) -> List[str]:
        if self.g2p is None:
            import g2pk

            self.g2p = g2pk.G2p()

        phones = list(
            self.g2p(
                text,
                descriptive=self.descritive,
                group_vowels=self.group_vowels,
                to_syl=self.to_syl,
            )
        )
        if self.no_space:
            # remove space which represents word serapater
            phones = list(filter(lambda s: s != " ", phones))

        if self.explicit_space:
            # replace space as explicit space symbol
            phones = list(map(lambda s: s if s != " " else self.space_symbol, phones))
        return phones


class Jaso:
    """
    A class for converting Korean text into Jamo characters.

    This class takes Korean text as input and converts it into its
    corresponding Jamo characters. It also provides options to handle
    spaces in the output.

    Attributes:
        PUNC (str): A string of punctuation characters.
        SPACE (str): A string representing a space character.
        JAMO_LEADS (str): A string of Jamo leading characters.
        JAMO_VOWELS (str): A string of Jamo vowel characters.
        JAMO_TAILS (str): A string of Jamo tail characters.
        VALID_CHARS (str): A string of valid characters, including Jamo
            characters, punctuation, and spaces.

    Args:
        space_symbol (str): The symbol to use for spaces in the output.
            Defaults to a regular space.
        no_space (bool): If True, spaces will be removed from the output.

    Examples:
        >>> jaso = Jaso(space_symbol="<space>", no_space=False)
        >>> jaso("안녕하세요")
        ['ᄋ', 'ᅡ', 'ᄂ', 'ᅣ', 'ᄉ', 'ᅥ', 'ᄒ', 'ᅡ', 'ᄋ', 'ᅭ']

        >>> jaso_no_space = Jaso(no_space=True)
        >>> jaso_no_space("안녕하세요")
        ['ᄋ', 'ᅡ', 'ᄂ', 'ᅣ', 'ᄉ', 'ᅥ', 'ᄒ', 'ᅡ', 'ᄋ', 'ᅭ']
    """

    PUNC = "!'(),-.:;?"
    SPACE = " "

    JAMO_LEADS = "".join([chr(_) for _ in range(0x1100, 0x1113)])
    JAMO_VOWELS = "".join([chr(_) for _ in range(0x1161, 0x1176)])
    JAMO_TAILS = "".join([chr(_) for _ in range(0x11A8, 0x11C3)])

    VALID_CHARS = JAMO_LEADS + JAMO_VOWELS + JAMO_TAILS + PUNC + SPACE

    def __init__(self, space_symbol=" ", no_space=False):
        self.space_symbol = space_symbol
        self.no_space = no_space

    def _text_to_jaso(self, line: str) -> List[str]:
        jasos = list(jamo.hangul_to_jamo(line))
        return jasos

    def _remove_non_korean_characters(self, tokens):
        new_tokens = [token for token in tokens if token in self.VALID_CHARS]
        return new_tokens

    def __call__(self, text) -> List[str]:
        graphemes = [x for x in self._text_to_jaso(text)]
        graphemes = self._remove_non_korean_characters(graphemes)

        if self.no_space:
            graphemes = list(filter(lambda s: s != " ", graphemes))
        else:
            graphemes = [x if x != " " else self.space_symbol for x in graphemes]
        return graphemes


class Phonemizer:
    """
        Phonemizer module for various languages.

    This is a wrapper module for the [phonemizer library, which provides
    phonemization capabilities for different languages. You can define
    various g2p (grapheme-to-phoneme) modules by specifying options for
    the phonemizer.

    See available options:
        https://github.com/bootphon/phonemizer/blob/master/phonemizer/phonemize.py#L32

    See also:
        https://github.com/bootphon/phonemizer

    Attributes:
        backend (str): The backend to use for phonemization.
        word_separator (Optional[str]): Custom word separator.
        syllable_separator (Optional[str]): Custom syllable separator.
        phone_separator (Optional[str]): Custom phone separator (default is " ").
        strip (bool): Whether to strip whitespace from the output.
        split_by_single_token (bool): Whether to split the output by single tokens.

    Args:
        backend (str): The backend for phonemization (e.g., "espeak").
        word_separator (Optional[str]): Custom word separator.
        syllable_separator (Optional[str]): Custom syllable separator.
        phone_separator (Optional[str]): Custom phone separator.
        strip (bool): Whether to strip whitespace from the output.
        split_by_single_token (bool): Whether to split the output by single tokens.
        **phonemizer_kwargs: Additional keyword arguments for phonemizer.

    Examples:
        >>> phonemizer = Phonemizer(backend='espeak')
        >>> phonemes = phonemizer("Hello, world!")
        >>> print(phonemes)
        ['h', 'ɛ', 'l', 'oʊ', ' ', 'w', 'ɜ', 'r', 'l', 'd']
    """

    def __init__(
        self,
        backend,
        word_separator: Optional[str] = None,
        syllable_separator: Optional[str] = None,
        phone_separator: Optional[str] = " ",
        strip=False,
        split_by_single_token: bool = False,
        **phonemizer_kwargs,
    ):
        # delayed import
        from phonemizer.backend import BACKENDS
        from phonemizer.separator import Separator

        self.separator = Separator(
            word=word_separator,
            syllable=syllable_separator,
            phone=phone_separator,
        )

        # define logger to suppress the warning in phonemizer
        logger = logging.getLogger("phonemizer")
        logger.setLevel(logging.ERROR)
        self.phonemizer = BACKENDS[backend](
            **phonemizer_kwargs,
            logger=logger,
        )
        self.strip = strip
        self.split_by_single_token = split_by_single_token

    def __call__(self, text) -> List[str]:
        tokens = self.phonemizer.phonemize(
            [text],
            separator=self.separator,
            strip=self.strip,
            njobs=1,
        )[0]
        if not self.split_by_single_token:
            return tokens.split()
        else:
            # "a: ab" -> ["a", ":", "<space>",  "a", "b"]
            # TODO(kan-bayashi): space replacement should be dealt in PhonemeTokenizer
            return [c.replace(" ", "<space>") for c in tokens]


class IsG2p:  # pylint: disable=too-few-public-methods
    """
        Minimal wrapper for https://github.com/grammatek/ice-g2p

    The g2p module uses a Bi-LSTM model along with
    a pronunciation dictionary to generate phonemization.
    Unfortunately, it does not support multi-thread phonemization as of yet.

    Attributes:
        dialect (str): The dialect to use for phonemization (default: "standard").
        syllabify (bool): Whether to syllabify the output (default: True).
        word_sep (str): The separator for words (default: ",").
        use_dict (bool): Whether to use a pronunciation dictionary (default: True).

    Args:
        dialect (str): The dialect for phonemization.
        syllabify (bool): Flag to enable syllabification.
        word_sep (str): Separator used for words.
        use_dict (bool): Flag to enable dictionary usage.

    Returns:
        List[str]: A list of phonemes generated from the input text.

    Examples:
        >>> g2p = IsG2p()
        >>> phonemes = g2p("example text")
        >>> print(phonemes)
        ['ɪ', 'g', 'z', 'æ', 'm', 'p', 'əl', 't', 'ɛ', 'k', 's']
    """

    def __init__(
        self,
        dialect: str = "standard",
        syllabify: bool = True,
        word_sep: str = ",",
        use_dict: bool = True,
    ):
        self.dialect = dialect
        self.syllabify = syllabify
        self.use_dict = use_dict
        from ice_g2p.transcriber import Transcriber

        self.transcriber = Transcriber(
            use_dict=self.use_dict,
            syllab_symbol=".",
            stress_label=True,
            word_sep=word_sep,
            lang_detect=True,
        )

    def __call__(self, text) -> List[str]:
        return self.transcriber.transcribe(text).split()


class PhonemeTokenizer(AbsTokenizer):
    """
    A tokenizer that converts text into phonemes using various G2P methods.

    This class is designed to handle text-to-phoneme (G2P) conversion for
    different languages and dialects. It supports multiple G2P backends
    and allows customization for handling non-linguistic symbols.

    Attributes:
        g2p_type (str): The type of G2P method to use for phoneme conversion.
        space_symbol (str): The symbol used to represent spaces in tokenized
            output.
        non_linguistic_symbols (set): A set of symbols to handle separately
            during tokenization.
        remove_non_linguistic_symbols (bool): Flag to determine whether to
            remove non-linguistic symbols from the output.

    Args:
        g2p_type (Union[None, str]): The G2P method to use. If None, a
            simple space-based tokenizer is used.
        non_linguistic_symbols (Union[None, Path, str, Iterable[str]]): A
            collection of non-linguistic symbols to handle.
        space_symbol (str): Symbol to use for spaces in tokenized output.
            Default is "<space>".
        remove_non_linguistic_symbols (bool): Whether to remove non-linguistic
            symbols from the output. Default is False.

    Raises:
        NotImplementedError: If an unsupported G2P type is provided.

    Examples:
        >>> tokenizer = PhonemeTokenizer(g2p_type="g2p_en")
        >>> tokens = tokenizer.text2tokens("Hello, world!")
        >>> print(tokens)
        ['HH', 'AH', 'L', 'OW', ',', 'W', 'ER', 'L', 'D', '!']

        >>> tokenizer = PhonemeTokenizer(g2p_type="g2pk",
        ...                                non_linguistic_symbols=["!"])
        >>> tokens = tokenizer.text2tokens("Hello! World!")
        >>> print(tokens)
        ['HH', 'AH', 'L', 'OW', '!', 'W', 'ER', 'L', 'D', '!']

    Note:
        The G2P methods used are dependent on the installation of
        corresponding libraries (e.g., g2p_en, g2pk, etc.). Make sure to
        install the necessary packages to utilize specific G2P methods.
    """

    @typechecked
    def __init__(
        self,
        g2p_type: Union[None, str],
        non_linguistic_symbols: Union[None, Path, str, Iterable[str]] = None,
        space_symbol: str = "<space>",
        remove_non_linguistic_symbols: bool = False,
    ):
        if g2p_type is None:
            self.g2p = split_by_space
        elif g2p_type == "g2p_en":
            self.g2p = G2p_en(no_space=False)
        elif g2p_type == "g2p_en_no_space":
            self.g2p = G2p_en(no_space=True)
        elif g2p_type == "pyopenjtalk":
            self.g2p = pyopenjtalk_g2p
        elif g2p_type == "pyopenjtalk_kana":
            self.g2p = pyopenjtalk_g2p_kana
        elif g2p_type == "pyopenjtalk_accent":
            self.g2p = pyopenjtalk_g2p_accent
        elif g2p_type == "pyopenjtalk_accent_with_pause":
            self.g2p = pyopenjtalk_g2p_accent_with_pause
        elif g2p_type == "pyopenjtalk_prosody":
            self.g2p = pyopenjtalk_g2p_prosody
        elif g2p_type == "pypinyin_g2p":
            self.g2p = pypinyin_g2p
        elif g2p_type == "pypinyin_g2p_phone":
            self.g2p = pypinyin_g2p_phone
        elif g2p_type == "pypinyin_g2p_phone_without_prosody":
            self.g2p = pypinyin_g2p_phone_without_prosody
        elif g2p_type == "espeak_ng_arabic":
            self.g2p = Phonemizer(
                language="ar",
                backend="espeak",
                with_stress=True,
                preserve_punctuation=True,
            )
        elif g2p_type == "espeak_ng_german":
            self.g2p = Phonemizer(
                language="de",
                backend="espeak",
                with_stress=True,
                preserve_punctuation=True,
            )
        elif g2p_type == "espeak_ng_french":
            self.g2p = Phonemizer(
                language="fr-fr",
                backend="espeak",
                with_stress=True,
                preserve_punctuation=True,
            )
        elif g2p_type == "espeak_ng_spanish":
            self.g2p = Phonemizer(
                language="es",
                backend="espeak",
                with_stress=True,
                preserve_punctuation=True,
            )
        elif g2p_type == "espeak_ng_russian":
            self.g2p = Phonemizer(
                language="ru",
                backend="espeak",
                with_stress=True,
                preserve_punctuation=True,
            )
        elif g2p_type == "espeak_ng_greek":
            self.g2p = Phonemizer(
                language="el",
                backend="espeak",
                with_stress=True,
                preserve_punctuation=True,
            )
        elif g2p_type == "espeak_ng_finnish":
            self.g2p = Phonemizer(
                language="fi",
                backend="espeak",
                with_stress=True,
                preserve_punctuation=True,
            )
        elif g2p_type == "espeak_ng_hungarian":
            self.g2p = Phonemizer(
                language="hu",
                backend="espeak",
                with_stress=True,
                preserve_punctuation=True,
            )
        elif g2p_type == "espeak_ng_dutch":
            self.g2p = Phonemizer(
                language="nl",
                backend="espeak",
                with_stress=True,
                preserve_punctuation=True,
            )
        elif g2p_type == "espeak_ng_hindi":
            self.g2p = Phonemizer(
                language="hi",
                backend="espeak",
                with_stress=True,
                preserve_punctuation=True,
            )
        elif g2p_type == "espeak_ng_italian":
            self.g2p = Phonemizer(
                language="it",
                backend="espeak",
                with_stress=True,
                preserve_punctuation=True,
            )
        elif g2p_type == "espeak_ng_polish":
            self.g2p = Phonemizer(
                language="pl",
                backend="espeak",
                with_stress=True,
                preserve_punctuation=True,
            )
        elif g2p_type == "g2pk":
            self.g2p = G2pk(no_space=False)
        elif g2p_type == "g2pk_no_space":
            self.g2p = G2pk(no_space=True)
        elif g2p_type == "g2pk_explicit_space":
            self.g2p = G2pk(explicit_space=True, space_symbol=space_symbol)
        elif g2p_type == "espeak_ng_english_us_vits":
            # VITS official implementation-like processing
            # Reference: https://github.com/jaywalnut310/vits
            self.g2p = Phonemizer(
                language="en-us",
                backend="espeak",
                with_stress=True,
                preserve_punctuation=True,
                strip=True,
                word_separator=" ",
                phone_separator="",
                split_by_single_token=True,
            )
        elif g2p_type == "korean_jaso":
            self.g2p = Jaso(space_symbol=space_symbol, no_space=False)
        elif g2p_type == "korean_jaso_no_space":
            self.g2p = Jaso(no_space=True)
        elif g2p_type == "g2p_is":
            self.g2p = IsG2p()
        elif g2p_type == "g2p_is_north":
            self.g2p = IsG2p(dialect="north")
        else:
            raise NotImplementedError(f"Not supported: g2p_type={g2p_type}")

        self.g2p_type = g2p_type
        self.space_symbol = space_symbol
        if non_linguistic_symbols is None:
            self.non_linguistic_symbols = set()
        elif isinstance(non_linguistic_symbols, (Path, str)):
            non_linguistic_symbols = Path(non_linguistic_symbols)
            try:
                with non_linguistic_symbols.open("r", encoding="utf-8") as f:
                    self.non_linguistic_symbols = set(line.rstrip() for line in f)
            except FileNotFoundError:
                warnings.warn(f"{non_linguistic_symbols} doesn't exist.")
                self.non_linguistic_symbols = set()
        else:
            self.non_linguistic_symbols = set(non_linguistic_symbols)
        self.remove_non_linguistic_symbols = remove_non_linguistic_symbols

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f'g2p_type="{self.g2p_type}", '
            f'space_symbol="{self.space_symbol}", '
            f'non_linguistic_symbols="{self.non_linguistic_symbols}"'
            ")"
        )

    def text2tokens(self, line: str) -> List[str]:
        """
                Converts input text to a list of tokens (phonemes).

        This method processes the input string `line` by extracting any non-linguistic
        symbols specified during the initialization of the `PhonemeTokenizer` and
        then applying the configured G2P (grapheme-to-phoneme) model to convert the
        remaining text into phonemes.

        Attributes:
            non_linguistic_symbols (set): A set of non-linguistic symbols to be
                recognized and handled during tokenization.
            remove_non_linguistic_symbols (bool): Flag indicating whether to
                remove non-linguistic symbols from the output tokens.

        Args:
            line (str): The input text string to be tokenized.

        Returns:
            List[str]: A list of tokens (phonemes) generated from the input text.

        Examples:
            >>> from phoneme_tokenizer import PhonemeTokenizer
            >>> tokenizer = PhonemeTokenizer(g2p_type="g2p_en")
            >>> tokenizer.text2tokens("Hello, world!")
            ['H', 'ə', 'l', 'oʊ', ' ', 'w', 'ɜ', 'r', 'l', 'd', '!']

        Note:
            The method processes the input line in a loop, checking for
            non-linguistic symbols at the beginning of the line. If found,
            it appends the symbol to the token list (if not set to remove)
            and continues processing the rest of the line. After handling
            all symbols, it applies the G2P model to the remaining text.
        """
        tokens = []
        while len(line) != 0:
            for w in self.non_linguistic_symbols:
                if line.startswith(w):
                    if not self.remove_non_linguistic_symbols:
                        tokens.append(line[: len(w)])
                    line = line[len(w) :]
                    break
            else:
                t = line[0]
                tokens.append(t)
                line = line[1:]

        line = "".join(tokens)
        tokens = self.g2p(line)
        return tokens

    def tokens2text(self, tokens: Iterable[str]) -> str:
        """
            Tokenizes text into phonemes using various g2p methods.

        This class serves as a tokenizer that converts text into phoneme tokens
        based on the specified g2p (grapheme-to-phoneme) method. It supports
        multiple g2p implementations, allowing for flexibility in phoneme
        generation for different languages and dialects.

        Attributes:
            g2p_type (Union[None, str]): The type of g2p method to use.
            space_symbol (str): The symbol to use for spaces in tokens.
            non_linguistic_symbols (set): A set of non-linguistic symbols to handle.
            remove_non_linguistic_symbols (bool): Whether to remove non-linguistic symbols.

        Args:
            g2p_type (Union[None, str]): The g2p method to be used.
            non_linguistic_symbols (Union[None, Path, str, Iterable[str]]):
                Symbols that are not linguistic in nature.
            space_symbol (str): The symbol used to represent spaces in tokens.
            remove_non_linguistic_symbols (bool): Whether to remove non-linguistic
                symbols from the output.

        Raises:
            NotImplementedError: If the specified g2p_type is not supported.

        Examples:
            >>> from espnet2.text.phoneme_tokenizer import PhonemeTokenizer
            >>> tokenizer = PhonemeTokenizer(g2p_type="g2p_en")
            >>> tokens = tokenizer.text2tokens("Hello, world!")
            >>> print(tokens)
            ['H', 'ə', 'l', 'oʊ', ',', ' ', 'w', 'ɜ', 'r', 'l', 'd', '!']

        Note:
            The tokenizer's behavior can be modified by adjusting the
            `non_linguistic_symbols` and `remove_non_linguistic_symbols`
            attributes.

        Todo:
            - Extend support for additional g2p methods.
            - Improve handling of edge cases in text-to-token conversion.
        """
        # phoneme type is not invertible
        return "".join(tokens)

    def text2tokens_svs(self, syllable: str) -> List[str]:
        """
        Converts a given syllable into its corresponding phonetic tokens.

        This method handles specific syllables by returning predefined token
        mappings from a custom dictionary. If the provided syllable is not in
        the dictionary, it defaults to using the general g2p (grapheme-to-phoneme)
        conversion method.

        Note:
            If needed, the `customed_dic` can be modified to include additional
            mappings as required.

        Args:
            syllable (str): The input syllable to be converted into phonetic
            tokens.

        Returns:
            List[str]: A list of phonetic tokens corresponding to the input
            syllable.

        Examples:
            >>> tokenizer = PhonemeTokenizer(g2p_type="pyopenjtalk")
            >>> tokenizer.text2tokens_svs("は")
            ['h', 'a']
            >>> tokenizer.text2tokens_svs("シ")
            ['sh', 'I']
            >>> tokenizer.text2tokens_svs("くぁ")
            ['k', 'w', 'a']
        """
        # Note(Yuning): fix syllabel2phoneme mismatch
        # If needed, customed_dic can be changed into extra input
        customed_dic = {
            "へ": ["h", "e"],
            "は": ["h", "a"],
            "シ": ["sh", "I"],
            "ヴぁ": ["v", "a"],
            "ヴぃ": ["v", "i"],
            "ヴぇ": ["v", "e"],
            "ヴぉ": ["v", "o"],
            "でぇ": ["dy", "e"],
            "くぁ": ["k", "w", "a"],
            "くぃ": ["k", "w", "i"],
            "くぅ": ["k", "w", "u"],
            "くぇ": ["k", "w", "e"],
            "くぉ": ["k", "w", "o"],
            "ぐぁ": ["g", "w", "a"],
            "ぐぃ": ["g", "w", "i"],
            "ぐぅ": ["g", "w", "u"],
            "ぐぇ": ["g", "w", "e"],
            "ぐぉ": ["g", "w", "o"],
            "くぉっ": ["k", "w", "o", "cl"],
        }
        tokens = self.g2p(syllable)
        if syllable in customed_dic:
            tokens = customed_dic[syllable]
        return tokens
