from abc import abstractmethod, ABC
from pathlib import Path
from typing import Union, Dict, List, Sequence

import numpy as np
import sentencepiece as spm
from typeguard import check_argument_types


class AbsTextConverter(ABC):
    @abstractmethod
    def __call__(self, line: str) -> np.ndarray:
        raise NotImplementedError

    def inverse(self, integers: Union[np.ndarray, Sequence[int]]) -> str:
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
        >>> int_array = converter(line)

    """
    def __init__(self, model: Union[Path, str]):
        assert check_argument_types()
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(model))

    def __call__(self, line: str) -> np.ndarray:
        ids = self.sp.EncodeAsIds(line)
        return np.array(ids)

    def inverse(self, integers: Union[np.ndarray, Sequence[int]]) -> str:
        if isinstance(integers, np.ndarray):
            if integers.dtype.kind != 'i':
                raise ValueError(
                    f'Must be int array: but got {integers.dtype}')
            integers = integers.tolist()
        return self.sp.DecodeIds(integers)


class Text2Words(AbsTextConverter):
    """

    Examples:
        >>> converter = Text2Words('word_list')
        >>> line = 'Hello! I am Naoyuki Kamo!!!'
        >>> int_array = converter(line)

    """
    def __init__(self, token_list: Union[Path, str, Sequence[str]],
                 unk_symbol: str = '<unk>',
                 delimiter: str = None,
                 ):
        assert check_argument_types()
        self.delimiter = delimiter
        self.unk_symbol = unk_symbol

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

    def __call__(self, line: str) -> np.ndarray:
        tokens = line.strip(self.delimiter)
        return np.fromiter((self.token2id.get(t, self.unk_id) for t in tokens),
                           dtype=np.int64)

    def inverse(self, integers: Union[np.ndarray, Sequence[int]]) -> str:
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
        >>> int_array = converter(line)

    """
    def __init__(self, token_list: Union[Path, str, Sequence[str]],
                 unk_symbol: str = '<unk>',
                 ):
        assert check_argument_types()
        token_list = Path(token_list)
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

    def __call__(self, line: str) -> np.ndarray:
        return np.fromiter((self.token2id.get(t, self.unk_id) for t in line),
                           dtype=np.int64)

    def inverse(self, integers: Union[np.ndarray, Sequence[int]]) -> str:
        return ''.join([self.id2token[i] for i in integers])

