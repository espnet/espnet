from pathlib import Path
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union
import warnings

import g2p_en
from typeguard import check_argument_types

from espnet2.text.abs_tokenizer import AbsTokenizer


def split_by_space(text) -> List[str]:
    return text.split(" ")


def pyopenjtalk_g2p(text) -> List[str]:
    import pyopenjtalk

    # phones is a str object separated by space
    phones = pyopenjtalk.g2p(text, kana=False)
    phones = phones.split(" ")
    return phones


def pyopenjtalk_g2p_accent(text) -> List[str]:
    import pyopenjtalk
    import re

    phones = []
    for labels in pyopenjtalk.run_frontend(text)[1]:
        p = re.findall(r"\-(.*?)\+.*?\/A:([0-9\-]+).*?\/F:.*?_([0-9])", labels)
        if len(p) == 1:
            phones += [p[0][0], p[0][2], p[0][1]]
    return phones


def pyopenjtalk_g2p_accent_with_pause(text) -> List[str]:
    import pyopenjtalk
    import re

    phones = []
    for labels in pyopenjtalk.run_frontend(text)[1]:
        if labels.split("-")[1].split("+")[0] == "pau":
            phones += ["pau"]
            continue
        p = re.findall(r"\-(.*?)\+.*?\/A:([0-9\-]+).*?\/F:.*?_([0-9])", labels)
        if len(p) == 1:
            phones += [p[0][0], p[0][2], p[0][1]]
    return phones


def pyopenjtalk_g2p_kana(text) -> List[str]:
    import pyopenjtalk

    kanas = pyopenjtalk.g2p(text, kana=True)
    return list(kanas)


def pypinyin_g2p(text) -> List[str]:
    from pypinyin import pinyin
    from pypinyin import Style

    phones = [phone[0] for phone in pinyin(text, style=Style.TONE3)]
    return phones


def pypinyin_g2p_phone(text) -> List[str]:
    from pypinyin import pinyin
    from pypinyin import Style
    from pypinyin.style._utils import get_finals
    from pypinyin.style._utils import get_initials

    phones = [
        p
        for phone in pinyin(text, style=Style.TONE3)
        for p in [
            get_initials(phone[0], strict=True),
            get_finals(phone[0], strict=True),
        ]
        if len(p) != 0
    ]
    return phones


class G2p_en:
    """On behalf of g2p_en.G2p.

    g2p_en.G2p isn't pickalable and it can't be copied to the other processes
    via multiprocessing module.
    As a workaround, g2p_en.G2p is instantiated upon calling this class.

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


class Phonemizer:
    """Phonemizer module for various languages.

    This is wrapper module of https://github.com/bootphon/phonemizer.
    You can define various g2p modules by specifying options for phonemizer.

    See available options:
        https://github.com/bootphon/phonemizer/blob/master/phonemizer/phonemize.py#L32

    """

    def __init__(
        self,
        word_separator: Optional[str] = None,
        syllable_separator: Optional[str] = None,
        **phonemize_kwargs,
    ):
        # delayed import
        from phonemizer import phonemize
        from phonemizer.separator import Separator

        self.phonemize = phonemize
        self.separator = Separator(
            word=word_separator, syllable=syllable_separator, phone=" "
        )
        self.phonemize_kwargs = phonemize_kwargs

    def __call__(self, text) -> List[str]:
        return self.phonemize(
            text,
            separator=self.separator,
            **self.phonemize_kwargs,
        ).split()


class PhonemeTokenizer(AbsTokenizer):
    def __init__(
        self,
        g2p_type: Union[None, str],
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        space_symbol: str = "<space>",
        remove_non_linguistic_symbols: bool = False,
    ):
        assert check_argument_types()
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
        elif g2p_type == "pypinyin_g2p":
            self.g2p = pypinyin_g2p
        elif g2p_type == "pypinyin_g2p_phone":
            self.g2p = pypinyin_g2p_phone
        elif g2p_type == "espeak_ng_arabic":
            self.g2p = Phonemizer(language="ar", backend="espeak", with_stress=True)
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
            f")"
        )

    def text2tokens(self, line: str) -> List[str]:
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
        # phoneme type is not invertible
        return "".join(tokens)
