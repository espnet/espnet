from typing import Iterable
from typing import List

from typeguard import check_argument_types

from espnet2.text.abs_tokenizer import AbsTokenizer


class WordTokenizer(AbsTokenizer):
    def __init__(self, delimiter: str = None):
        assert check_argument_types()
        self.delimiter = delimiter

    def __repr__(self):
        return f'{self.__class__.__name__}(delimiter="{self.delimiter}")'

    def text2tokens(self, line: str) -> List[str]:
        return line.split(self.delimiter)

    def tokens2text(self, tokens: Iterable[str]) -> str:
        if self.delimiter is None:
            delimiter = " "
        else:
            delimiter = self.delimiter
        return delimiter.join(tokens)
