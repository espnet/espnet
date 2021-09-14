from pathlib import Path
from typing import Iterable
from typing import List
from typing import Union
import warnings

from typeguard import check_argument_types

from espnet2.text.abs_tokenizer import AbsTokenizer

import jamo ## jamo==0.4.1

PUNC = '!\'(),-.:;?'
SPACE = '<space>'

JAMO_LEADS = "".join([chr(_) for _ in range(0x1100, 0x1113)])
JAMO_VOWELS = "".join([chr(_) for _ in range(0x1161, 0x1176)])
JAMO_TAILS = "".join([chr(_) for _ in range(0x11A8, 0x11C3)])

VALID_CHARS = JAMO_LEADS + JAMO_VOWELS + JAMO_TAILS + PUNC + SPACE

class JasoTokenizer(AbsTokenizer):
    def __init__(
        self,
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        space_symbol: str = "<space>",
        remove_non_linguistic_symbols: bool = False,
    ):
        assert check_argument_types()
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
            f'space_symbol="{self.space_symbol}"'
            f'non_linguistic_symbols="{self.non_linguistic_symbols}"'
            f")"
        )

    def _text_to_jaso(self, line : str) -> List[str]:
        jasos = list(jamo.hangul_to_jamo(line))
        return jasos

    def _remove_non_korean_characters(self, tokens):
        new_tokens = [token for token in tokens if token in VALID_CHARS]
        return new_tokens

    def text2tokens(self, line: str) -> List[str]:
        tokens = []

        new_line_tokens = []
        if self.remove_non_linguistic_symbols:
            for t in line:
                if not t in self.non_linguistic_symbols:
                    new_line_tokens.append(t)
        else:
            new_line_tokens = [x for x in line]
        new_line = ''.join(new_line_tokens)

        tokens = [x if x != ' ' else '<space>' for x in self._text_to_jaso(new_line)]
        tokens = self._remove_non_korean_characters(tokens)
        return tokens

if __name__ == '__main__':
    
    tokenizer = JasoTokenizer()

    text = "나는 학교에 간다."
    tokens = tokenizer.text2tokens(text)
    print(tokens)