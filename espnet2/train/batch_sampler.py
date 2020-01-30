from __future__ import annotations

from abc import ABC
from abc import abstractmethod
import logging
from typing import Iterator
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

from torch.utils.data import Sampler
from typeguard import check_argument_types
from typeguard import check_return_type

from espnet2.utils.fileio import load_num_sequence_text
from espnet2.utils.fileio import read_2column_text


def build_batch_sampler(
    type: str,
    batch_size: int,
    shape_files: Union[Tuple[str, ...], List[str]],
    max_lengths: Sequence[int] = (),
    sort_in_batch: str = "descending",
    sort_batch: str = "ascending",
) -> AbsSampler:
    """Helper function to instantiate BatchSampler

    Args:
        type: mini-batch type. "constant", "seq", "bin", or, "frame"
        batch_size: The mini-batch size
        shape_files: Text files describing the length and dimension
            of each features. e.g. uttA 1330,80
        max_lengths: Used for "seq" mode
        sort_in_batch:
        sort_batch:
    """
    assert check_argument_types()

    if type == "const_no_sort":
        retval = ConstantBatchSampler(batch_size=batch_size, key_file=shape_files[0])

    elif type == "const":
        retval = ConstantSortedBatchSampler(
            batch_size=batch_size,
            shape_file=shape_files[0],
            sort_in_batch=sort_in_batch,
            sort_batch=sort_batch,
        )

    elif type == "seq":
        if len(max_lengths) != len(shape_files):
            raise ValueError(
                f"The number of max_lengths must be equal to "
                f"the number of shape_files: "
                f"{len(max_lengths)} != {len(shape_files)}"
            )
        retval = SequenceBatchSampler(
            batch_size=batch_size,
            shape_files=shape_files,
            max_lengths=max_lengths,
            sort_in_batch=sort_in_batch,
            sort_batch=sort_batch,
        )

    elif type == "bin":
        raise NotImplementedError

    elif type == "frame":
        raise NotImplementedError

    else:
        raise RuntimeError(f"Not supported: {type}")
    assert check_return_type(retval)
    return retval


class AbsSampler(Sampler, ABC):
    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        raise NotImplementedError


class ConstantSortedBatchSampler(AbsSampler):
    """BatchSampler with sorted samples by length.

    Args:
        batch_size:
        shape_file:
        sort_in_batch: 'descending', 'ascending' or None.
        sort_batch:
    """

    def __init__(
        self,
        batch_size: int,
        shape_file: str,
        sort_in_batch: str = "descending",
        sort_batch: str = "ascending",
    ):
        assert check_argument_types()
        self.batch_size = batch_size
        self.shape_file = shape_file
        self.sort_in_batch = sort_in_batch
        self.sort_batch = sort_batch

        # utt2shape: (Length, ...)
        #    uttA 100,...
        #    uttB 201,...
        utt2shape = load_num_sequence_text(shape_file, loader_type="csv_int")
        if sort_in_batch == "descending":
            # Sort samples in descending order (required by RNN)
            keys = sorted(utt2shape, key=lambda k: -utt2shape[k][0])
        elif sort_in_batch == "ascending":
            # Sort samples in ascending order
            keys = sorted(utt2shape, key=lambda k: utt2shape[k][0])
        else:
            raise ValueError(
                f"sort_in_batch must be either one of "
                f"ascending, descending, or None: {sort_in_batch}"
            )

        # batch_list: List[Tuple[str, ...]]
        self.batch_list = [
            tuple(keys[i : i + batch_size]) for i in range(0, len(keys), batch_size)
        ]

        if len(self.batch_list) == 0:
            logging.warning(f"{shape_file} is empty")

        if sort_in_batch != sort_batch:
            if sort_batch not in ("ascending", "descending"):
                raise ValueError(
                    f"sort_batch must be ascending or descending: {sort_batch}"
                )
            self.batch_list.reverse()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_size={self.batch_size}, "
            f"shape_file={self.shape_file}, "
            f"sort_in_batch={self.sort_in_batch}, "
            f"sort_batch={self.sort_batch})"
        )

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        for batch in self.batch_list:
            yield batch


class ConstantBatchSampler(AbsSampler):
    """BatchSampler with constant batch-size.

    Any sorting is not done in this class,
    so no length information is required,
    This class is convenient for decoding mode,
    or not seq2seq learning e.g. classification.

    Args:
        batch_size:
        key_file:
    """

    def __init__(self, batch_size: int, key_file: str):
        assert check_argument_types()
        self.batch_size = batch_size
        self.key_file = key_file

        # utt2shape:
        #    uttA <anything is o.k>
        #    uttB <anything is o.k>
        utt2any = read_2column_text(key_file)
        if len(utt2any) == 0:
            logging.warning(f"{key_file} is empty")
        # In this case the, the first column in only used
        self.keys = list(utt2any)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_size={self.batch_size}, "
            f"key_file={self.key_file}, "
        )

    def __len__(self):
        return len(self.keys)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        # batch_list: List[Tuple[str, ...]]
        batch_list = [
            tuple(self.keys[i : i + self.batch_size])
            for i in range(0, len(self.keys), self.batch_size)
        ]
        for batch in batch_list:
            yield batch


class SequenceBatchSampler(AbsSampler):
    def __init__(
        self,
        batch_size: int,
        shape_files: Union[Tuple[str, ...], List[str]],
        max_lengths: Sequence[int],
        min_batch_size: int = 1,
        sort_in_batch: str = "descending",
        sort_batch: str = "ascending",
    ):
        assert check_argument_types()
        self.batch_size = batch_size
        self.shape_files = shape_files
        self.sort_in_batch = sort_in_batch
        self.sort_batch = sort_batch

        # utt2shape: (Length, ...)
        #    uttA 100,...
        #    uttB 201,...
        utt2shapes = [
            load_num_sequence_text(s, loader_type="csv_int") for s in shape_files
        ]

        first_utt2shape = utt2shapes[0]
        for s, d in zip(shape_files, utt2shapes):
            if set(d) != set(first_utt2shape):
                raise RuntimeError(
                    f"keys are mismatched between {s} != {shape_files[0]}"
                )

        # Sort samples in ascending order
        # (shape order should be like (Length, Dim))
        keys = sorted(first_utt2shape, key=lambda k: first_utt2shape[k][0])

        start = 0
        self.batch_list = []
        while True:
            k = keys[start]
            factor = max(int(d[k][0] / m) for d, m in zip(utt2shapes, max_lengths))
            bs = max(min_batch_size, int(batch_size / (1 + factor)))
            minibatch_keys = keys[start : start + bs]
            if sort_in_batch == "descending":
                minibatch_keys.reverse()
            elif sort_in_batch == "ascending":
                pass
            else:
                raise ValueError(
                    f"sort_in_batch must be ascending or descending: {sort_in_batch}"
                )
            self.batch_list.append(tuple(minibatch_keys))
            start += bs
            if start >= len(keys):
                break

        if sort_batch == "ascending":
            pass
        elif sort_batch == "descending":
            self.batch_list.reverse()
        else:
            raise ValueError(
                f"sort_batch must be ascending or descending: {sort_batch}"
            )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_size={self.batch_size}, "
            f"shape_files={self.shape_files}, "
            f"sort_in_batch={self.sort_in_batch}, "
            f"sort_batch={self.sort_batch})"
        )

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        for batch in self.batch_list:
            yield batch
