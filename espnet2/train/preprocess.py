from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Union, Sequence

import numpy as np
from typeguard import check_argument_types, check_return_type

from espnet2.utils.text_converter import build_text_converter


class AbsPreprocessor(ABC):
    def __init__(self, train_or_eval: str):
        self.train_or_eval = train_or_eval

    @abstractmethod
    def __call__(self, data: Dict[str, Union[str, np.ndarray]]):
        raise NotImplementedError


class CommonPreprocessor(AbsPreprocessor):
    def __init__(self,
                 train_or_eval: str,
                 token_type: str = None,
                 model_or_token_list: Union[Path, str, Sequence[str]] = None,
                 unk_symbol: str = '<unk>',
                 delimiter: str = None,
                 ):
        super().__init__(train_or_eval)
        self.train_or_eval = train_or_eval

        if token_type is not None:
            if model_or_token_list is None:
                raise ValueError('model_or_token_list is required '
                                 'if token_type is not None')

            self.converter = build_text_converter(
                token_type=token_type, model_or_token_list=model_or_token_list,
                unk_symbol=unk_symbol, delimiter=delimiter)

    def __call__(self, data: Dict[str, Union[str, np.ndarray]]):
        assert check_argument_types()

        if 'feats' in data:
            # Nothing now: candidates:
            # - STFT
            # - Fbank
            # - CMVN
            # - Data augmentation
            pass

        if 'text' in data:
            text = data['text']
            text_int_array = self.converter(text)
            data['text'] = text_int_array

        assert check_return_type(data)
        return data
