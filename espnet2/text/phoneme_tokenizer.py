from pathlib import Path
from typing import Iterable
from typing import List
from typing import Union

import g2p_en
from typeguard import check_argument_types

from espnet2.text.abs_tokenizer import AbsTokenizer


def pyopenjtalk_g2p(text) -> List[str]:
    import pyopenjtalk

    # phones is a str object separated by space
    phones = pyopenjtalk.g2p(text, kana=False)
    phones = phones.split(" ")
    return phones


def pyopenjtalk_g2p_kana(text) -> List[str]:
    import pyopenjtalk

    kanas = pyopenjtalk.g2p(text, kana=True)
    return list(kanas)


class PhonemeTokenizer(AbsTokenizer):
    def __init__(
        self,
        g2p_type: str,
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        space_symbol: str = "<space>",
        remove_non_linguistic_symbols: bool = False,
    ):
        assert check_argument_types()
        if g2p_type == "g2p_en":
            self.g2p = g2p_en.G2p()
        elif g2p_type == "pyopenjtalk":
            self.g2p = pyopenjtalk_g2p
        elif g2p_type == "pyopenjtalk_kana":
            self.g2p = pyopenjtalk_g2p_kana
        else:
            raise NotImplementedError(f"Not supported: g2p_type={g2p_type}")

        self.g2p_type = g2p_type
        self.space_symbol = space_symbol
        if non_linguistic_symbols is None:
            self.non_linguistic_symbols = set()
        elif isinstance(non_linguistic_symbols, (Path, str)):
            non_linguistic_symbols = Path(non_linguistic_symbols)
            with non_linguistic_symbols.open("r", encoding="utf-8") as f:
                self.non_linguistic_symbols = set(line.rstrip() for line in f)
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
