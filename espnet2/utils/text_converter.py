from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Dict
from typing import List
from typing import Union
from typing import Iterable

import numpy as np
import sentencepiece as spm
from typeguard import check_argument_types


class AbsTextConverter(ABC):
    def __init__(self, token_list: Union[Path, str, Iterable[str]],
                 unk_symbol: str = "<unk>",
                 ):
        assert check_argument_types()

        if isinstance(token_list, (Path, str)):
            token_list = Path(token_list)
            self.token_list_repr = str(token_list)
            self.token_list: List[str] = []

            with token_list.open("r") as f:
                for idx, line in enumerate(f):
                    line = line.rstrip()
                    self.token_list.append(line)

        else:
            self.token_list: List[str] = list(token_list)
            self.token_list_repr = ""
            for i, t in enumerate(self.token_list):
                if i == 3:
                    break
                self.token_list_repr += f"{t}, "
            self.token_list_repr += f"... (NVocab={(len(self.token_list))})"

        self.token2id: Dict[str, int] = {}
        for i, t in enumerate(self.token_list):
            if t in self.token2id:
                raise RuntimeError(f'Symbol "{t}" is duplicated')
            self.token2id[t] = i

        self.unk_symbol = unk_symbol
        if self.unk_symbol not in self.token2id:
            raise RuntimeError(
                f'Unknown symbol "{unk_symbol}" '
                f"doesn't exist in the token_list"
            )
        self.unk_id = self.token2id[self.unk_symbol]

    def get_num_vocaburary_size(self) -> int:
        return len(self.token_list)

    def ids2tokens(self, integers: Union[np.ndarray, Iterable[int]]) \
            -> List[str]:
        if isinstance(integers, np.ndarray) and integers.ndim != 1:
            raise ValueError(f"Must be 1 dim ndarray, but got {integers.ndim}")
        return [self.token_list[i] for i in integers]

    def tokens2ids(self, tokens: Iterable[str]) -> List[int]:
        return [self.token2id.get(i, self.unk_id) for i in tokens]

    def text2ids(self, line: str) -> List[int]:
        tokens = self.text2tokens(line)
        return self.tokens2ids(tokens)

    def ids2text(self, integers: Union[np.ndarray, Iterable[int]]) -> str:
        tokens = self.ids2tokens(integers)
        return self.tokens2text(tokens)

    @abstractmethod
    def text2tokens(self, line: str) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def tokens2text(self, tokens: Iterable[str]) -> str:
        raise NotImplementedError


def build_text_converter(
    token_type: str,
    token_list: Union[Path, str, Iterable[str]],
    bpemodel: Union[Path, str, Iterable[str]] = None,
    unk_symbol: str = "<unk>",
    delimiter: str = None,
) -> AbsTextConverter:
    """A helper function to instantiate Tokenizer"""
    assert check_argument_types()
    if token_type == "bpe":
        if bpemodel is None:
            raise ValueError('bpemodel is required if token_type = "bpeh"')
        return Text2SentencepiecesConverter(token_list, bpemodel)

    elif token_type == "word":
        return Text2WordsConverter(
            token_list, unk_symbol=unk_symbol, delimiter=delimiter
        )

    elif token_type == "char":
        return Text2CharsConverter(token_list, unk_symbol=unk_symbol)

    else:
        raise ValueError(
            f"token_mode must be one of bpe, word, or char: " f"{token_type}"
        )


class Text2SentencepiecesConverter(AbsTextConverter):
    """


    Text2Sentencepieces uses
    "DecodePieces" instead of "DecodeIds" always
    because we may not keep the consistency of symbol-id
    between Sentencepiece and DNN training.


    Examples:
        >>> converter = Text2SentencepiecesConverter('bpe_token.list', 'bpe.model')
        >>> line = 'Hello! I am Naoyuki Kamo!!!'
        >>> int_array = converter.text2ids(line)

    """

    def __init__(self, token_list: Union[Path, str, Iterable[str]],
                 model: Union[Path, str],
                 ):
        assert check_argument_types()
        super().__init__(token_list=token_list)
        self.model = Path(model)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(model))

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'model="{self.model}", '
                f'token_list="{self.token_list_repr}"'
                f')')

    def text2tokens(self, line: str) -> List[str]:
        return self.sp.EncodeAsPieces(line)

    def tokens2text(self, tokens: Iterable[str]) -> str:
        return self.sp.DecodePieces(list(tokens))


class Text2WordsConverter(AbsTextConverter):
    """

    Examples:
        >>> converter = Text2WordsConverter('word.list')
        >>> line = 'Hello! I am Naoyuki Kamo!!!'
        >>> int_array = converter.text2ids(line)

    """

    def __init__(
        self,
        token_list: Union[Path, str, Iterable[str]],
        unk_symbol: str = "<unk>",
        delimiter: str = None,
    ):
        assert check_argument_types()
        super().__init__(token_list=token_list, unk_symbol=unk_symbol)
        self.delimiter = delimiter

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f'token_list="{self.token_list_repr}", '
            f'unk_symbol="{self.unk_symbol}", '
            f'delimiter="{self.delimiter}"'
            f")"
        )

    def text2tokens(self, line: str) -> List[str]:
        return line.split(self.delimiter)

    def tokens2text(self, tokens: Iterable[str]) -> str:
        if self.delimiter is None:
            delimiter = " "
        else:
            delimiter = self.delimiter
        return delimiter.join(tokens)


class Text2CharsConverter(AbsTextConverter):
    """

    Examples:
        >>> converter = Text2CharsConverter('char.list')
        >>> line = 'Hello! I am Naoyuki Kamo!!!'
        >>> int_array = converter.text2ids(line)

    """

    def __init__(
        self,
        token_list: Union[Path, str, Iterable[str]],
        unk_symbol: str = "<unk>",
    ):
        assert check_argument_types()
        super().__init__(token_list=token_list, unk_symbol=unk_symbol)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f'token_list="{self.token_list_repr}", '
            f'unk_symbol="{self.unk_symbol}"'
            f")"
        )

    def tokens2text(self, tokens: Iterable[str]) -> str:
        return "".join(tokens)

    def text2tokens(self, line: str) -> List[str]:
        return list(line)
