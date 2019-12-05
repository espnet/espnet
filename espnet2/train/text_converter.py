from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Dict
from typing import List
from typing import Sequence
from typing import Union

import numpy as np
import sentencepiece as spm
from typeguard import check_argument_types
from typing import Iterable


class AbsTextConverter(ABC):
    @abstractmethod
    def text2ids(self, line: str) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def tokens2text(self, tokens: Iterable[str]) -> str:
        raise NotImplementedError

    @abstractmethod
    def ids2text(self, integers: Union[np.ndarray, Sequence[int]]) -> str:
        raise NotImplementedError


def build_text_converter(token_type: str,
                         model_or_token_list: Union[Path, str, Sequence[str]],
                         unk_symbol: str = '<unk>',
                         delimiter: str = None,
                         ) -> AbsTextConverter:
    """A helper function to instantiate Tokenizer"""
    assert check_argument_types()
    if token_type == 'bpe':
        return Text2Sentencepieces(model_or_token_list)
    elif token_type == 'word':
        return Text2Words(model_or_token_list, unk_symbol=unk_symbol,
                          delimiter=delimiter)
    elif token_type == 'char':
        return Text2Chars(model_or_token_list, unk_symbol=unk_symbol)
    else:
        raise ValueError(f'token_mode must be one of bpe, word, or char: '
                         f'{token_type}')


class Text2Sentencepieces(AbsTextConverter):
    """

    Examples:
        >>> converter = Text2Sentencepieces('bpe.model')
        >>> line = 'Hello! I am Naoyuki Kamo!!!'
        >>> int_array = converter.text2ids(line)

    """
    def __init__(self, model: Union[Path, str]):
        assert check_argument_types()
        self.model = Path(model)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(model))

    def __repr__(self):
        return f'{self.__class__.__name__}(model="{self.model}")'

    def text2ids(self, line: str) -> np.ndarray:
        ids = self.sp.EncodeAsIds(line)
        return np.array(ids)

    def tokens2text(self, tokens: Iterable[str]) -> str:
        text = self.sp.DecodePieces(list(tokens))
        return text

    def ids2text(self, integers: Union[np.ndarray, Sequence[int]]) -> str:
        if isinstance(integers, np.ndarray):
            if integers.dtype.kind != 'i':
                raise ValueError(
                    f'Must be int array: but got {integers.dtype}')
            integers = integers.tolist()
        text = self.sp.DecodeIds(integers)
        return text


class Text2Words(AbsTextConverter):
    """

    Examples:
        >>> converter = Text2Words('word_list')
        >>> line = 'Hello! I am Naoyuki Kamo!!!'
        >>> int_array = converter.text2ids(line)

    """
    def __init__(self, token_list: Union[Path, str, Sequence[str]],
                 unk_symbol: str = '<unk>',
                 delimiter: str = None,
                 ):
        assert check_argument_types()
        self.delimiter = delimiter
        self.unk_symbol = unk_symbol
        token_list = Path(token_list)
        self.token_path = token_list

        if isinstance(token_list, (Path, str)):
            self.token_list: List[str] = []
            self.token2id: Dict[str, int] = {}
            token_list = Path(token_list)
            with token_list.open('r') as f:
                for idx, line in enumerate(f):
                    line = line.rstrip()
                    self.token_list.append(line)
                    self.token2id[line] = idx

        else:
            self.token_list: List[str] = list(token_list)
            self.token2id = {t: i for i, t in enumerate(self.token_list)}
        self.id2token = {i: t for i, t in self.id2token}

        if self.unk_symbol not in self.token2id:
            raise RuntimeError(f'Unknown symbol "{unk_symbol}" '
                               f'doesn\'t exist in {token_list}')
        self.unk_id = self.token2id[self.unk_symbol]

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'token_list="{self.token_path}", '
                f'unk_symbol="{self.unk_symbol}", '
                f'delimiter="{self.delimiter}", '
                f')')

    def text2ids(self, line: str) -> np.ndarray:
        tokens = line.strip(self.delimiter)
        return np.fromiter((self.token2id.get(t, self.unk_id) for t in tokens),
                           dtype=np.int64)

    def tokens2text(self, tokens: Iterable[str]) -> str:
        return self.delimiter.join(tokens)

    def ids2text(self, integers: Union[np.ndarray, Sequence[int]]) -> str:
        if self.delimiter is None:
            delimiter = ' '
        else:
            delimiter = self.delimiter
        return delimiter.join([self.id2token[i] for i in integers])


class Text2Chars(AbsTextConverter):
    """

    Examples:
        >>> converter = Text2Chars('char_list')
        >>> line = 'Hello! I am Naoyuki Kamo!!!'
        >>> int_array = converter.text2ids(line)

    """
    def __init__(self, token_list: Union[Path, str, Sequence[str]],
                 unk_symbol: str = '<unk>',
                 ):
        assert check_argument_types()
        token_list = Path(token_list)
        self.token_path = token_list

        self.unk_symbol = unk_symbol
        if isinstance(token_list, (Path, str)):
            self.token_list: List[str] = []
            self.token2id: Dict[str, int] = {}
            with token_list.open('r') as f:
                for idx, line in enumerate(f):
                    line = line.rstrip()
                    self.token_list.append(line)
                    self.token2id[line] = idx

        else:
            self.token_list: List[str] = list(token_list)
        self.id2token = {i: t for i, t in self.id2token}

        if self.unk_symbol not in self.token2id:
            raise RuntimeError(f'Unknown symbol "{unk_symbol}" '
                               f"doesn't exist in {token_list}")
        self.unk_id = self.token2id[self.unk_symbol]

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'token_list="{self.token_path}", '
                f'unk_symbol="{self.unk_symbol}", '
                f')')

    def text2ids(self, line: str) -> np.ndarray:
        return np.fromiter((self.token2id.get(t, self.unk_id) for t in line),
                           dtype=np.int64)

    def tokens2text(self, tokens: Iterable[str]) -> str:
        return ''.join(tokens)

    def ids2text(self, integers: Union[np.ndarray, Sequence[int]]) -> str:
        return ''.join([self.id2token[i] for i in integers])
