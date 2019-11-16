from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Union, Tuple, List, Iterator

import numpy as np
from torch.utils.data import Sampler
from typeguard import typechecked

from espnet2.utils.fileio import load_num_sequence_text


@typechecked
def create_batch_sampler(
        type: str,
        batch_size: int,
        shape_files: Union[Tuple[str, ...], List[str]],
        shuffle: bool = False,
        sort_in_batch_descending: bool = True,
        sort_descending: bool = False) -> AbsSampler:
    """Helper function to instantiate BatchSampler

    Args:
        batch_size: The mini-batch size
        type: type indicates the type of batch sampler
        shape_files: Text files describing the length and dimension
            of each features. e.g. uttA 1330,80
        shuffle: If False, the batch are sorted by ascending order.
    """
    if type == 'constant':
        return ConstantBatchSampler(
            batch_size=batch_size, shape_file=shape_files[0],
            shuffle=shuffle, sort_in_batch_descending=sort_in_batch_descending,
            sort_descending=sort_descending)

    elif type == 'batchbin':
        return BatchbinBatchSampler(
            batch_size=batch_size, shape_files=shape_files, shuffle=shuffle)

    elif type == 'batchbin':
        return BatchbinBatchSampler(
            batch_size=batch_size, shape_files=shape_files, shuffle=shuffle)

    else:
        raise RuntimeError(f'Not supported: {type}')


class AbsSampler(ABC, Sampler):
    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[str]]:
        raise NotImplementedError


class ConstantBatchSampler(AbsSampler):
    @typechecked
    def __init__(self,
                 batch_size: int,
                 shape_file: str,
                 shuffle: bool = False,
                 sort_in_batch_descending: bool = True,
                 sort_descending: bool = False,
                 ):
        self.shuffle = shuffle
        self.batch_size = batch_size

        # utt2shape: (Length, Dim)
        #    uttA 100,80
        #    uttg 201,80
        utt2shape = load_num_sequence_text(shape_file, loader_type='csv_int')
        if sort_in_batch_descending:
            # Sort samples in descending order (required by RNN)
            keys = sorted(utt2shape, key=lambda k: -utt2shape[k][0])
        else:
            # Sort samples in ascending order
            keys = sorted(utt2shape, key=lambda k: utt2shape[k][0])

        # batch_list: List[Tuple[str, ...]]
        self.batch_list = \
            [tuple(keys[i:i + batch_size])
             for i in range(0, int(np.ceil(len(keys) / batch_size)),
                            batch_size)]

        if len(self.batch_list) == 0:
            logging.warning(f'{shape_file} is empty')

        if self.shuffle:
            np.random.shuffle(self.batch_list)
        else:
            if sort_in_batch_descending != sort_descending:
                # Sort batches in ascending order
                self.batch_list = self.batch_list[::-1]

    def __len__(self):
        raise len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str]]:
        for batch in self.batch_list:
            yield batch


class BatchbinBatchSampler(AbsSampler):
    @typechecked
    def __init__(self,
                 batch_size: int,
                 shape_files: Union[Tuple[str, ...], List[str]],
                 shuffle: bool = False):
        raise NotImplementedError
        if self.shuffle:
            np.random.shuffle(self.batch_list)

    def __len__(self):
        raise len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str]]:
        for batch in self.batch_list:
            yield batch