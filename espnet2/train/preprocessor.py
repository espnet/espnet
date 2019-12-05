from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Dict
from typing import Sequence
from typing import Union

import numpy as np
from typeguard import check_argument_types
from typeguard import check_return_type

from espnet2.utils.text_converter import build_text_converter
from espnet2.utils.fileio import DatadirWriter


class AbsPreprocessor(ABC):
    def __init__(self, train: bool):
        self.train = train

    @abstractmethod
    def __call__(self, uid: str, data: Dict[str, Union[str, np.ndarray]]):
        raise NotImplementedError


class CommonPreprocessor(AbsPreprocessor):
    def __init__(self,
                 train: bool,
                 token_type: str = None,
                 model_or_token_list: Union[Path, str, Sequence[str]] = None,
                 unk_symbol: str = '<unk>',
                 delimiter: str = None,
                 speech_name: str = 'speech',
                 text_name: str = 'text',
                 output_dir: Union[Path, str] = None,
                 ):
        super().__init__(train)
        self.train = train
        self.speech_name = speech_name
        self.text_name = text_name

        if token_type is not None:
            if model_or_token_list is None:
                raise ValueError('model_or_token_list is required '
                                 'if token_type is not None')

            self.text_converter = build_text_converter(
                token_type=token_type, model_or_token_list=model_or_token_list,
                unk_symbol=unk_symbol, delimiter=delimiter)
        else:
            self.text_converter = None

        if output_dir is not None:
            self.dir_writer = DatadirWriter(output_dir)
        else:
            self.dir_writer = None

    def __call__(self, uid: str, data: Dict[str, Union[str, np.ndarray]]):
        assert check_argument_types()

        if self.speech_name in data:
            # Nothing now: candidates:
            # - STFT
            # - Fbank
            # - CMVN
            # - Data augmentation
            pass

        if self.text_name in data and self.text_converter is not None:
            text = data[self.text_name]
            text_int_array = self.text_converter.text2ids(text)
            data[self.text_name] = text_int_array

        # TODO(kamo): I couldn't find clear way to realize this
        # [Option] Derive the shape
        if self.dir_writer is not None:
            for k, v in data.items():
                shape = ','.join(map(str, v.shape))
                self.dir_writer[k + '_shape'][uid] = shape

        assert check_return_type(data)
        return data
