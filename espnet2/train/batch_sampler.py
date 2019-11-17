from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Union, Tuple, List, Iterator, Optional, Sequence

import numpy as np
from torch.utils.data import Sampler
from typeguard import typechecked

from espnet2.utils.fileio import load_num_sequence_text, read_2column_text


@typechecked
def create_batch_sampler(
        type: str,
        batch_size: int,
        shape_files: Union[Tuple[str, ...], List[str]],
        max_lengths: Sequence[int] = (),
        shuffle: bool = False,
        sort_in_batch: Optional[str] = 'descending',
        sort_batch: str = 'ascending') -> AbsSampler:
    """Helper function to instantiate BatchSampler

    Args:
        type: mini-batch type. "constant", "seq", "bin", or, "frame"
        batch_size: The mini-batch size
        shape_files: Text files describing the length and dimension
            of each features. e.g. uttA 1330,80
        max_lengths: Used for "seq" mode
        shuffle: If False, the batch are sorted by ascending order.
        sort_in_batch:
        sort_batch:
    """
    if type == 'const':
        return ConstantBatchSampler(
            batch_size=batch_size, shape_file=shape_files[0],
            shuffle=shuffle,
            sort_in_batch=sort_in_batch,
            sort_batch=sort_batch)

    elif type == 'seq':
        if len(max_lengths) != len(shape_files):
            raise ValueError(f'The number of max_lengths must be equal to '
                             f'the number of shape_files: '
                             f'{len(max_lengths)} != {len(shape_files)}')
        return SequenceBatchSampler(
            batch_size=batch_size, shape_files=shape_files,
            max_lengths=max_lengths,
            shuffle=shuffle,
            sort_in_batch=sort_in_batch,
            sort_batch=sort_batch)

    elif type == 'bin':
        raise NotImplementedError

    elif type == 'frame':
        raise NotImplementedError

    else:
        raise RuntimeError(f'Not supported: {type}')


class AbsSampler(ABC, Sampler):
    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        raise NotImplementedError


class ConstantBatchSampler(AbsSampler):
    @typechecked
    def __init__(self,
                 batch_size: int,
                 shape_file: str,
                 shuffle: bool = False,
                 sort_in_batch: Optional[str] = 'descending',
                 sort_batch: str = 'ascending',
                 ):
        self.shuffle = shuffle
        self.batch_size = batch_size

        # utt2shape: (Length, Dim)
        #    uttA 100,80
        #    uttg 201,80
        if sort_in_batch is None:
            utt2shape = read_2column_text(shape_file)
            # In this case the, the first column in only used
            keys = list(utt2shape)
            if self.shuffle:
                np.random.shuffle(keys)
        else:
            utt2shape = \
                load_num_sequence_text(shape_file, loader_type='csv_int')
            if sort_in_batch == 'descending':
                # Sort samples in descending order (required by RNN)
                keys = sorted(utt2shape, key=lambda k: -utt2shape[k][0])
            elif sort_in_batch == 'ascending':
                # Sort samples in ascending order
                keys = sorted(utt2shape, key=lambda k: utt2shape[k][0])
            else:
                raise ValueError(
                    f'sort_in_batch must be either one of '
                    f'ascending, descending, or None: {sort_in_batch}')

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
            if sort_in_batch is not None and sort_in_batch != sort_batch:
                if sort_batch not in ('ascending', 'descending'):
                    raise ValueError(
                        f'sort_batch must be ascending or '
                        f'descending: {sort_batch}')
                self.batch_list.reverse()

    def __len__(self):
        raise len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        for batch in self.batch_list:
            yield batch


class SequenceBatchSampler(AbsSampler):
    @typechecked
    def __init__(self,
                 batch_size: int,
                 shape_files: Union[Tuple[str, ...], List[str]],
                 max_lengths: Sequence[int],
                 min_batch_size: int = 1,
                 shuffle: bool = False,
                 sort_in_batch: str = 'descending',
                 sort_batch: str = 'ascending',
                 ):
        self.shuffle = shuffle
        self.batch_size = batch_size

        utt2shapes = [load_num_sequence_text(s, loader_type='csv_int')
                      for s in shape_files]
        for s, d in zip(shape_files, utt2shapes):
            if set(d) != set(utt2shapes[0]):
                raise RuntimeError(
                    f'keys are mismathed between {s} != {shape_files[0]}')

        # Sort samples in ascending order
        keys = sorted(utt2shapes[0], key=lambda k: utt2shapes[0][k])

        start = 0
        self.batch_list = []
        while True:
            k = keys[start]
            factor = max(int(d[k][0] / m) for d, m
                         in zip(utt2shapes. max_lengths))
            bs = max(min_batch_size, int(batch_size / (1 + factor)))
            minibatch_keys = keys[start:start + bs]
            if sort_in_batch == 'descending':
                minibatch_keys.reverse()
            elif sort_in_batch == 'ascending':
                pass
            else:
                raise ValueError(f'sort_in_batch must be ascending or '
                                 f'descending: {sort_in_batch}')
            self.batch_list.append(tuple(minibatch_keys))
            start += bs
            if start >= len(keys):
                break

        if self.shuffle:
            np.random.shuffle(self.batch_list)
        else:
            if sort_batch == 'ascending':
                pass
            elif sort_batch == 'decending':
                self.batch_list.reverse()
            else:
                raise ValueError(f'sort_batch must be ascending or '
                                 f'descending: {sort_batch}')

    def __len__(self):
        raise len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        for batch in self.batch_list:
            yield batch
