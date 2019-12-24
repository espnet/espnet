from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Iterable
from typing import List
from typing import Union

import sentencepiece as spm
from typeguard import check_argument_types


class AbsTokenizer(ABC):
    @abstractmethod
    def text2tokens(self, line: str) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def tokens2text(self, tokens: Iterable[str]) -> str:
        raise NotImplementedError


def build_tokenizer(
    token_type: str,
    bpemodel: Union[Path, str, Iterable[str]] = None,
    non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
    remove_non_linguistic_symbols: bool = False,
    space_symbol: str = "<space>",
    delimiter: str = None,
) -> AbsTokenizer:
    """A helper function to instantiate Tokenizer"""
    assert check_argument_types()
    if token_type == "bpe":
        if bpemodel is None:
            raise ValueError('bpemodel is required if token_type = "bpe"')
        return SentencepiecesTokenizer(bpemodel)

    elif token_type == "word":
        return WordTokenizer(delimiter=delimiter)

    elif token_type == "char":
        return CharTokenizer(
            non_linguistic_symbols=non_linguistic_symbols,
            space_symbol=space_symbol,
            remove_non_linguistic_symbols=remove_non_linguistic_symbols,
        )

    else:
        raise ValueError(
            f"token_mode must be one of bpe, word, or char: " f"{token_type}"
        )


class SentencepiecesTokenizer(AbsTokenizer):
    def __init__(self, model: Union[Path, str]):
        assert check_argument_types()
        self.model = Path(model)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(model))

    def __repr__(self):
        return f'{self.__class__.__name__}(model="{self.model}")'

    def text2tokens(self, line: str) -> List[str]:
        return self.sp.EncodeAsPieces(line)

    def tokens2text(self, tokens: Iterable[str]) -> str:
        return self.sp.DecodePieces(list(tokens))


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


class CharTokenizer(AbsTokenizer):
    def __init__(
        self,
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        space_symbol: str = "<space>",
        remove_non_linguistic_symbols: bool = False,
    ):
        assert check_argument_types()
        self.space_symbol = space_symbol
        if non_linguistic_symbols is None:
            self.non_linguistic_symbols = []
        elif isinstance(non_linguistic_symbols, (Path, str)):
            non_linguistic_symbols = Path(non_linguistic_symbols)
            with non_linguistic_symbols.open("r") as f:
                self.non_linguistic_symbols = [line.rstrip() for line in f]
        else:
            self.non_linguistic_symbols = list(non_linguistic_symbols)
        self.remove_non_linguistic_symbols = remove_non_linguistic_symbols

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f'space_symbol="{self.space_symbol}"'
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
                if t == " ":
                    t = "<space>"
                tokens.append(t)
                line = line[1:]
        return tokens

    def tokens2text(self, tokens: Iterable[str]) -> str:
        tokens = [t if t != self.space_symbol else " " for t in tokens]
        return "".join(tokens)
